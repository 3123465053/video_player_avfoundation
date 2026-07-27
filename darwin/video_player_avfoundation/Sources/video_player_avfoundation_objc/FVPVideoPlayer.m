// Copyright 2013 The Flutter Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#import "./include/video_player_avfoundation_objc/FVPVideoPlayer.h"
#import "./include/video_player_avfoundation_objc/FVPVideoPlayer_Internal.h"

#import <AudioToolbox/AudioToolbox.h>
#import <GLKit/GLKit.h>
#import <MediaToolbox/MediaToolbox.h>
#import <math.h>
#import <pthread.h>

#import "./include/video_player_avfoundation_objc/AVAssetTrackUtils.h"

#define FVP_EQ_BAND_COUNT 10
#define FVP_EQ_MAX_CHANNELS 8

static void *timeRangeContext = &timeRangeContext;
static void *statusContext = &statusContext;
static void *playbackLikelyToKeepUpContext = &playbackLikelyToKeepUpContext;
static void *rateContext = &rateContext;

typedef struct {
  double b0;
  double b1;
  double b2;
  double a1;
  double a2;
} FVPBiquadCoefficients;

typedef struct {
  double z1;
  double z2;
} FVPBiquadState;

typedef struct {
  pthread_mutex_t mutex;
  bool enabled;
  bool supportsFloat32;
  double sampleRate;
  UInt32 channelCount;
  double outputGain;
  double gains[FVP_EQ_BAND_COUNT];
  FVPBiquadCoefficients coeffs[FVP_EQ_BAND_COUNT];
  FVPBiquadState states[FVP_EQ_BAND_COUNT][FVP_EQ_MAX_CHANNELS];
} FVPEqualizerContext;

static const double FVPBandFrequencies[FVP_EQ_BAND_COUNT] = {
    31.0, 62.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0};

static double FVPClampDouble(double value, double min, double max) {
  if (value < min) {
    return min;
  }
  if (value > max) {
    return max;
  }
  return value;
}

static void FVPClearEqualizerState(FVPEqualizerContext *context) {
  if (context == NULL) {
    return;
  }
  memset(context->states, 0, sizeof(context->states));
}

static void FVPRecalculateEqualizerCoefficients(FVPEqualizerContext *context) {
  if (context == NULL || context->sampleRate <= 0) {
    return;
  }

  double nyquist = context->sampleRate * 0.5;
  double maxBoost = 0.0;
  double totalBoost = 0.0;
  for (int band = 0; band < FVP_EQ_BAND_COUNT; band++) {
    double gain = FVPClampDouble(context->gains[band], -12.0, 12.0);
    if (gain > 0.0) {
      maxBoost = MAX(maxBoost, gain);
      totalBoost += gain;
    }
    if (fabs(gain) < 0.001) {
      context->coeffs[band] = (FVPBiquadCoefficients){1.0, 0.0, 0.0, 0.0, 0.0};
      continue;
    }

    double frequency = FVPClampDouble(FVPBandFrequencies[band], 20.0, nyquist * 0.95);
    double q = 0.85;
    double omega = 2.0 * M_PI * frequency / context->sampleRate;
    double sinOmega = sin(omega);
    double cosOmega = cos(omega);
    double alpha = sinOmega / (2.0 * q);
    double amplitude = pow(10.0, gain / 40.0);

    double b0 = 1.0 + alpha * amplitude;
    double b1 = -2.0 * cosOmega;
    double b2 = 1.0 - alpha * amplitude;
    double a0 = 1.0 + alpha / amplitude;
    double a1 = -2.0 * cosOmega;
    double a2 = 1.0 - alpha / amplitude;

    context->coeffs[band] =
        (FVPBiquadCoefficients){b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0};
  }

  double attenuationDb = MIN(10.0, maxBoost * 0.55 + totalBoost * 0.08);
  context->outputGain = pow(10.0, -attenuationDb / 20.0);
}

static double FVPSoftLimit(double value) {
  if (value > 0.98) {
    return 0.98 + 0.02 * tanh((value - 0.98) * 8.0);
  }
  if (value < -0.98) {
    return -0.98 + 0.02 * tanh((value + 0.98) * 8.0);
  }
  return value;
}

static float FVPProcessEqualizerSample(FVPEqualizerContext *context, UInt32 channel, float sample) {
  if (context == NULL || !context->enabled || context->sampleRate <= 0 ||
      channel >= FVP_EQ_MAX_CHANNELS) {
    return sample;
  }

  double value = sample;
  for (int band = 0; band < FVP_EQ_BAND_COUNT; band++) {
    FVPBiquadCoefficients c = context->coeffs[band];
    FVPBiquadState *state = &context->states[band][channel];
    double output = c.b0 * value + state->z1;
    state->z1 = c.b1 * value - c.a1 * output + state->z2;
    state->z2 = c.b2 * value - c.a2 * output;
    value = output;
  }

  value *= context->outputGain <= 0.0 ? 1.0 : context->outputGain;
  return (float)FVPClampDouble(FVPSoftLimit(value), -1.0, 1.0);
}

static void FVPEqualizerTapInit(MTAudioProcessingTapRef tap, void *clientInfo,
                                void **tapStorageOut) {
  *tapStorageOut = clientInfo;
}

static void FVPEqualizerTapFinalize(MTAudioProcessingTapRef tap) {}

static void FVPEqualizerTapPrepare(MTAudioProcessingTapRef tap, CMItemCount maxFrames,
                                   const AudioStreamBasicDescription *processingFormat) {
  FVPEqualizerContext *context =
      (FVPEqualizerContext *)MTAudioProcessingTapGetStorage(tap);
  if (context == NULL || processingFormat == NULL) {
    return;
  }

  pthread_mutex_lock(&context->mutex);
  context->sampleRate = processingFormat->mSampleRate;
  context->channelCount = processingFormat->mChannelsPerFrame;
  context->supportsFloat32 = processingFormat->mFormatID == kAudioFormatLinearPCM &&
                             (processingFormat->mFormatFlags & kAudioFormatFlagIsFloat) != 0 &&
                             processingFormat->mBitsPerChannel == 32 &&
                             processingFormat->mBytesPerFrame >= sizeof(float);
  FVPRecalculateEqualizerCoefficients(context);
  FVPClearEqualizerState(context);
  pthread_mutex_unlock(&context->mutex);
}

static void FVPEqualizerTapUnprepare(MTAudioProcessingTapRef tap) {}

static void FVPEqualizerTapProcess(MTAudioProcessingTapRef tap, CMItemCount numberFrames,
                                   MTAudioProcessingTapFlags flags,
                                   AudioBufferList *bufferListInOut,
                                   CMItemCount *numberFramesOut,
                                   MTAudioProcessingTapFlags *flagsOut) {
  OSStatus status = MTAudioProcessingTapGetSourceAudio(tap, numberFrames, bufferListInOut, flagsOut,
                                                       NULL, numberFramesOut);
  if (status != noErr || bufferListInOut == NULL || numberFramesOut == NULL ||
      *numberFramesOut <= 0) {
    return;
  }

  FVPEqualizerContext *context =
      (FVPEqualizerContext *)MTAudioProcessingTapGetStorage(tap);
  if (context == NULL || pthread_mutex_trylock(&context->mutex) != 0) {
    return;
  }
  if (!context->enabled || !context->supportsFloat32) {
    pthread_mutex_unlock(&context->mutex);
    return;
  }

  UInt32 frameCount = (UInt32)*numberFramesOut;
  BOOL nonInterleaved = bufferListInOut->mNumberBuffers > 1;

  for (UInt32 bufferIndex = 0; bufferIndex < bufferListInOut->mNumberBuffers; bufferIndex++) {
    AudioBuffer buffer = bufferListInOut->mBuffers[bufferIndex];
    if (buffer.mData == NULL) {
      continue;
    }

    UInt32 channelsInBuffer = nonInterleaved ? 1 : MAX(buffer.mNumberChannels, 1);
    UInt32 samplesInBuffer = frameCount * channelsInBuffer;
    if (buffer.mDataByteSize < samplesInBuffer * sizeof(float)) {
      continue;
    }

    float *samples = (float *)buffer.mData;
    for (UInt32 frame = 0; frame < frameCount; frame++) {
      for (UInt32 channel = 0; channel < channelsInBuffer; channel++) {
        UInt32 sampleIndex = frame * channelsInBuffer + channel;
        UInt32 channelIndex = nonInterleaved ? bufferIndex : channel;
        samples[sampleIndex] =
            FVPProcessEqualizerSample(context, channelIndex, samples[sampleIndex]);
      }
    }
  }
  pthread_mutex_unlock(&context->mutex);
}

/// Registers KVO observers on 'object' for each entry in 'observations', which must be a
/// dictionary mapping KVO keys to NSValue-wrapped context pointers.
///
/// This does not call any methods on 'observer', so is safe to call from 'observer's init.
static void FVPRegisterKeyValueObservers(NSObject *observer,
                                         NSDictionary<NSString *, NSValue *> *observations,
                                         NSObject *target) {
  // It is important not to use NSKeyValueObservingOptionInitial here, because that will cause
  // synchronous calls to 'observer', violating the requirement that this method does not call its
  // methods. If there are use cases for specific pieces of initial state, those should be handled
  // explicitly by the caller, rather than by adding initial-state KVO notifications here.
  for (NSString *key in observations) {
    [target addObserver:observer
             forKeyPath:key
                options:NSKeyValueObservingOptionNew
                context:observations[key].pointerValue];
  }
}

/// Registers KVO observers on 'object' for each entry in 'observations', which must be a
/// dictionary mapping KVO keys to NSValue-wrapped context pointers.
///
/// This should only be called to balance calls to FVPRegisterKeyValueObservers, as it is an
/// error to try to remove observers that are not currently set.
///
/// This does not call any methods on 'observer', so is safe to call from 'observer's dealloc.
static void FVPRemoveKeyValueObservers(NSObject *observer,
                                       NSDictionary<NSString *, NSValue *> *observations,
                                       NSObject *target) {
  for (NSString *key in observations) {
    [target removeObserver:observer forKeyPath:key];
  }
}

/// Returns a mapping of KVO keys to NSValue-wrapped observer context pointers for observations that
/// should be set for AVPlayer instances.
static NSDictionary<NSString *, NSValue *> *FVPGetPlayerObservations(void) {
  return @{
    @"rate" : [NSValue valueWithPointer:rateContext],
  };
}

/// Returns a mapping of KVO keys to NSValue-wrapped observer context pointers for observations that
/// should be set for AVPlayerItem instances.
static NSDictionary<NSString *, NSValue *> *FVPGetPlayerItemObservations(void) {
  return @{
    @"loadedTimeRanges" : [NSValue valueWithPointer:timeRangeContext],
    @"status" : [NSValue valueWithPointer:statusContext],
    @"playbackLikelyToKeepUp" : [NSValue valueWithPointer:playbackLikelyToKeepUpContext],
  };
}

@implementation FVPVideoPlayer {
  // Whether or not player and player item listeners have ever been registered.
  BOOL _listenersRegistered;
  NSObject<FVPAVPlayerItem> *_playerItem;
  FVPEqualizerContext *_equalizerContext;
  BOOL _equalizerEnabled;
}

- (instancetype)initWithPlayerItem:(NSObject<FVPAVPlayerItem> *)item
                         avFactory:(id<FVPAVFactory>)avFactory
                      viewProvider:(NSObject<FVPViewProvider> *)viewProvider {
  self = [super init];
  NSAssert(self, @"super init cannot be nil");

  _viewProvider = viewProvider;
  _playerItem = item;
  _equalizerContext = calloc(1, sizeof(FVPEqualizerContext));
  if (_equalizerContext != NULL) {
    pthread_mutex_init(&_equalizerContext->mutex, NULL);
    _equalizerContext->enabled = false;
    _equalizerContext->supportsFloat32 = false;
    _equalizerContext->outputGain = 1.0;
  }

  NSObject<FVPAVAsset> *asset = item.asset;
  void (^assetCompletionHandler)(void) = ^{
    if ([asset statusOfValueForKey:@"tracks" error:nil] == AVKeyValueStatusLoaded) {
      void (^processVideoTracks)(NSArray<AVAssetTrack *> *) = ^(NSArray<AVAssetTrack *> *tracks) {
        if ([tracks count] > 0) {
          AVAssetTrack *videoTrack = tracks[0];
          void (^trackCompletionHandler)(void) = ^{
            if (self->_disposed) return;
            if ([videoTrack statusOfValueForKey:@"preferredTransform"
                                          error:nil] == AVKeyValueStatusLoaded) {
              // Rotate the video by using a videoComposition and the preferredTransform
              self->_preferredTransform = FVPGetStandardizedTrackTransform(
                  videoTrack.preferredTransform, videoTrack.naturalSize);
              // Do not use video composition when it is not needed.
              if (CGAffineTransformIsIdentity(self->_preferredTransform)) {
                return;
              }
              // Note:
              // https://developer.apple.com/documentation/avfoundation/avplayeritem/1388818-videocomposition
              // Video composition can only be used with file-based media and is not supported for
              // use with media served using HTTP Live Streaming.
              AVMutableVideoComposition *videoComposition =
                  [self videoCompositionWithTransform:self->_preferredTransform
                                                asset:asset
                                           videoTrack:videoTrack];
              item.videoComposition = videoComposition;
            }
          };
          [videoTrack loadValuesAsynchronouslyForKeys:@[ @"preferredTransform" ]
                                    completionHandler:trackCompletionHandler];
        }
      };

      // Use the new async API on iOS 15.0+/macOS 12.0+, fall back to deprecated API on older
      // versions
      if (@available(iOS 15.0, macOS 12.0, *)) {
        [asset loadTracksWithMediaType:AVMediaTypeVideo
                     completionHandler:^(NSArray<AVAssetTrack *> *_Nullable tracks,
                                         NSError *_Nullable error) {
                       if (error == nil && tracks != nil) {
                         processVideoTracks(tracks);
                       } else if (error != nil) {
                         NSLog(@"Error loading tracks: %@", error);
                       }
                     }];
      } else {
        // For older OS versions, use the deprecated API with warning suppression
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        NSArray *tracks = [asset tracksWithMediaType:AVMediaTypeVideo];
#pragma clang diagnostic pop
        processVideoTracks(tracks);
      }
    }
  };

  _player = [avFactory playerWithPlayerItem:item];
  _player.actionAtItemEnd = AVPlayerActionAtItemEndNone;

  // Configure output. AVVideoColorPropertiesKey must be declared on the output settings (not on
  // pixel buffer attributes, where AVFoundation silently ignores it) so that HDR sources are
  // tone-mapped to BT.709 SDR for the Flutter texture.
  // See https://github.com/flutter/flutter/issues/91241
  NSDictionary *outputSettings = @{
    AVVideoColorPropertiesKey : @{
      AVVideoColorPrimariesKey : AVVideoColorPrimaries_ITU_R_709_2,
      AVVideoTransferFunctionKey : AVVideoTransferFunction_ITU_R_709_2,
      AVVideoYCbCrMatrixKey : AVVideoYCbCrMatrix_ITU_R_709_2,
    },
    (id)kCVPixelBufferPixelFormatTypeKey : @(kCVPixelFormatType_32BGRA),
    (id)kCVPixelBufferIOSurfacePropertiesKey : @{},
  };
  _pixelBufferSource = [avFactory videoOutputWithOutputSettings:outputSettings];

  [asset loadValuesAsynchronouslyForKeys:@[ @"tracks" ] completionHandler:assetCompletionHandler];

  return self;
}

- (void)dealloc {
  if (_listenersRegistered && !_disposed) {
    // If dispose was never called for some reason, remove observers to prevent crashes.
    FVPRemoveKeyValueObservers(self, FVPGetPlayerItemObservations(), _player.currentItem);
    FVPRemoveKeyValueObservers(self, FVPGetPlayerObservations(), _player);
  }
  if (_equalizerContext != NULL) {
    _playerItem.audioMix = nil;
    pthread_mutex_lock(&_equalizerContext->mutex);
    _equalizerContext->enabled = false;
    pthread_mutex_unlock(&_equalizerContext->mutex);
    pthread_mutex_destroy(&_equalizerContext->mutex);
    free(_equalizerContext);
  }
}

- (void)disposeWithError:(FlutterError *_Nullable *_Nonnull)error {
  // In some hot restart scenarios, dispose can be called twice, so no-op after the first time.
  if (_disposed) {
    return;
  }
  _disposed = YES;

  if (_listenersRegistered) {
    [[NSNotificationCenter defaultCenter] removeObserver:self];
    FVPRemoveKeyValueObservers(self, FVPGetPlayerItemObservations(), self.player.currentItem);
    FVPRemoveKeyValueObservers(self, FVPGetPlayerObservations(), self.player);
  }

  _playerItem.audioMix = nil;
  [self.player replaceCurrentItemWithPlayerItem:nil];

  if (_onDisposed) {
    _onDisposed();
  }
  [self.eventListener videoPlayerWasDisposed];
}

- (void)setEqualizerEnabled:(BOOL)enabled {
  BOOL needsAudioMixUpdate = (_equalizerEnabled != enabled) ||
                             (enabled && _playerItem.audioMix == nil);
  _equalizerEnabled = enabled;
  if (_equalizerContext != NULL) {
    pthread_mutex_lock(&_equalizerContext->mutex);
    _equalizerContext->enabled = enabled;
    if (needsAudioMixUpdate) {
      FVPClearEqualizerState(_equalizerContext);
    }
    pthread_mutex_unlock(&_equalizerContext->mutex);
  }
  if (needsAudioMixUpdate) {
    [self updateEqualizerAudioMix];
  }
}

- (void)setEqualizerBands:(NSArray<NSNumber *> *)gains {
  if (_equalizerContext == NULL) {
    return;
  }

  pthread_mutex_lock(&_equalizerContext->mutex);
  for (int index = 0; index < FVP_EQ_BAND_COUNT; index++) {
    double gain = 0.0;
    if (index < gains.count) {
      gain = gains[index].doubleValue;
    }
    _equalizerContext->gains[index] = FVPClampDouble(gain, -12.0, 12.0);
  }

  FVPRecalculateEqualizerCoefficients(_equalizerContext);
  pthread_mutex_unlock(&_equalizerContext->mutex);
  if (_equalizerEnabled && _playerItem.audioMix == nil) {
    [self updateEqualizerAudioMix];
  }
}

- (void)updateEqualizerAudioMix {
  if (!_equalizerEnabled || _playerItem == nil || _equalizerContext == NULL) {
    _playerItem.audioMix = nil;
    return;
  }

  NSObject<FVPAVAsset> *asset = _playerItem.asset;
  void (^applyAudioMix)(NSArray<AVAssetTrack *> *) = ^(NSArray<AVAssetTrack *> *tracks) {
    if (!self->_equalizerEnabled || self->_disposed || tracks.count == 0) {
      self->_playerItem.audioMix = nil;
      return;
    }

    MTAudioProcessingTapCallbacks callbacks;
    callbacks.version = kMTAudioProcessingTapCallbacksVersion_0;
    callbacks.clientInfo = self->_equalizerContext;
    callbacks.init = FVPEqualizerTapInit;
    callbacks.finalize = FVPEqualizerTapFinalize;
    callbacks.prepare = FVPEqualizerTapPrepare;
    callbacks.unprepare = FVPEqualizerTapUnprepare;
    callbacks.process = FVPEqualizerTapProcess;

    MTAudioProcessingTapRef tap = NULL;
    OSStatus status = MTAudioProcessingTapCreate(kCFAllocatorDefault, &callbacks,
                                                 kMTAudioProcessingTapCreationFlag_PostEffects,
                                                 &tap);
    if (status != noErr || tap == NULL) {
      self->_playerItem.audioMix = nil;
      return;
    }

    AVMutableAudioMixInputParameters *parameters =
        [AVMutableAudioMixInputParameters audioMixInputParametersWithTrack:tracks.firstObject];
    parameters.audioTapProcessor = tap;

    AVMutableAudioMix *audioMix = [AVMutableAudioMix audioMix];
    audioMix.inputParameters = @[ parameters ];
    self->_playerItem.audioMix = audioMix;
    CFRelease(tap);
  };

  if (@available(iOS 15.0, macOS 12.0, *)) {
    [asset loadTracksWithMediaType:AVMediaTypeAudio
                 completionHandler:^(NSArray<AVAssetTrack *> *_Nullable tracks,
                                     NSError *_Nullable error) {
                   if (error == nil && tracks != nil) {
                     dispatch_async(dispatch_get_main_queue(), ^{
                       applyAudioMix(tracks);
                     });
                   }
                 }];
  } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    NSArray<AVAssetTrack *> *tracks = [asset tracksWithMediaType:AVMediaTypeAudio];
#pragma clang diagnostic pop
    applyAudioMix(tracks);
  }
}

- (void)setEventListener:(NSObject<FVPVideoEventListener> *)eventListener {
  _eventListener = eventListener;
  // The first time an event listener is set, set up video event listeners to relay status changes
  // changes to the event listener.
  if (eventListener && !_listenersRegistered) {
    AVPlayerItem *item = self.player.currentItem;
    // If the item is already ready to play, ensure that the intialized event is sent first.
    [self reportStatusForPlayerItem:item];
    // Set up all necessary observers to report video events.
    FVPRegisterKeyValueObservers(self, FVPGetPlayerItemObservations(), item);
    FVPRegisterKeyValueObservers(self, FVPGetPlayerObservations(), _player);
    [[NSNotificationCenter defaultCenter] addObserver:self
                                             selector:@selector(itemDidPlayToEndTime:)
                                                 name:AVPlayerItemDidPlayToEndTimeNotification
                                               object:item];
    _listenersRegistered = YES;
  }
}

- (void)itemDidPlayToEndTime:(NSNotification *)notification {
  if (_isLooping) {
    AVPlayerItem *p = [notification object];
    [p seekToTime:kCMTimeZero completionHandler:nil];
  } else {
    [self.eventListener videoPlayerDidComplete];
  }
}

const int64_t TIME_UNSET = -9223372036854775807;

NS_INLINE int64_t FVPCMTimeToMillis(CMTime time) {
  // When CMTIME_IS_INDEFINITE return a value that matches TIME_UNSET from ExoPlayer2 on Android.
  // Fixes https://github.com/flutter/flutter/issues/48670
  if (CMTIME_IS_INDEFINITE(time)) return TIME_UNSET;
  if (time.timescale == 0) return 0;
  return time.value * 1000 / time.timescale;
}

NS_INLINE CGFloat radiansToDegrees(CGFloat radians) {
  // Input range [-pi, pi] or [-180, 180]
  CGFloat degrees = GLKMathRadiansToDegrees((float)radians);
  if (degrees < 0) {
    // Convert -90 to 270 and -180 to 180
    return degrees + 360;
  }
  // Output degrees in between [0, 360]
  return degrees;
};

- (AVMutableVideoComposition *)videoCompositionWithTransform:(CGAffineTransform)transform
                                                       asset:(NSObject<FVPAVAsset> *)asset
                                                  videoTrack:(AVAssetTrack *)videoTrack {
  AVMutableVideoCompositionInstruction *instruction =
      [AVMutableVideoCompositionInstruction videoCompositionInstruction];
  instruction.timeRange = CMTimeRangeMake(kCMTimeZero, asset.duration);
  AVMutableVideoCompositionLayerInstruction *layerInstruction =
      [AVMutableVideoCompositionLayerInstruction
          videoCompositionLayerInstructionWithAssetTrack:videoTrack];
  [layerInstruction setTransform:_preferredTransform atTime:kCMTimeZero];

  AVMutableVideoComposition *videoComposition = [AVMutableVideoComposition videoComposition];
  instruction.layerInstructions = @[ layerInstruction ];
  videoComposition.instructions = @[ instruction ];

  // If in portrait mode, switch the width and height of the video
  CGFloat width = videoTrack.naturalSize.width;
  CGFloat height = videoTrack.naturalSize.height;
  NSInteger rotationDegrees =
      (NSInteger)round(radiansToDegrees(atan2(_preferredTransform.b, _preferredTransform.a)));
  if (rotationDegrees == 90 || rotationDegrees == 270) {
    width = videoTrack.naturalSize.height;
    height = videoTrack.naturalSize.width;
  }
  videoComposition.renderSize = CGSizeMake(width, height);

  videoComposition.sourceTrackIDForFrameTiming = videoTrack.trackID;
  if (CMTIME_IS_VALID(videoTrack.minFrameDuration)) {
    videoComposition.frameDuration = videoTrack.minFrameDuration;
  } else {
    NSLog(@"Warning: videoTrack.minFrameDuration for input video is invalid, please report this to "
          @"https://github.com/flutter/flutter/issues with input video attached.");
    videoComposition.frameDuration = CMTimeMake(1, 30);
  }

  return videoComposition;
}

- (void)observeValueForKeyPath:(NSString *)path
                      ofObject:(id)object
                        change:(NSDictionary *)change
                       context:(void *)context {
  if (context == timeRangeContext) {
    NSMutableArray<NSArray<NSNumber *> *> *values = [[NSMutableArray alloc] init];
    for (NSValue *rangeValue in [object loadedTimeRanges]) {
      CMTimeRange range = [rangeValue CMTimeRangeValue];
      [values addObject:@[
        @(FVPCMTimeToMillis(range.start)),
        @(FVPCMTimeToMillis(range.duration)),
      ]];
    }
    [self.eventListener videoPlayerDidUpdateBufferRegions:values];
  } else if (context == statusContext) {
    AVPlayerItem *item = (AVPlayerItem *)object;
    [self reportStatusForPlayerItem:item];
  } else if (context == playbackLikelyToKeepUpContext) {
    [self updatePlayingState];
    if ([[_player currentItem] isPlaybackLikelyToKeepUp]) {
      [self.eventListener videoPlayerDidEndBuffering];
    } else {
      [self.eventListener videoPlayerDidStartBuffering];
    }
  } else if (context == rateContext) {
    // Important: Make sure to cast the object to AVPlayer when observing the rate property,
    // as it is not available in AVPlayerItem.
    AVPlayer *player = (AVPlayer *)object;
    [self.eventListener videoPlayerDidSetPlaying:(player.rate > 0)];
  }
}

- (void)reportStatusForPlayerItem:(AVPlayerItem *)item {
  NSAssert(self.eventListener,
           @"reportStatusForPlayerItem was called when the event listener was not set.");
  switch (item.status) {
    case AVPlayerItemStatusFailed:
      [self sendFailedToLoadVideoEvent];
      break;
    case AVPlayerItemStatusUnknown:
      break;
    case AVPlayerItemStatusReadyToPlay:
      if (!_isInitialized) {
        [item addOutput:self.pixelBufferSource.videoOutput];
        [self reportInitialized];
        [self updatePlayingState];
      }
      break;
  }
}

- (void)updatePlayingState {
  if (!_isInitialized) {
    return;
  }
  if (_isPlaying) {
    // Calling play is the same as setting the rate to 1.0 (or to defaultRate depending on iOS
    // version) so last set playback speed must be set here if any instead.
    // https://github.com/flutter/flutter/issues/71264
    // https://github.com/flutter/flutter/issues/73643
    if (_targetPlaybackSpeed) {
      [self updateRate];
    } else {
      [_player play];
    }
  } else {
    [_player pause];
  }
}

/// Synchronizes the player's playback rate with targetPlaybackSpeed, constrained by the playback
/// rate capabilities of the player's current item.
- (void)updateRate {
  // See https://developer.apple.com/library/archive/qa/qa1772/_index.html for an explanation of
  // these checks.
  // If status is not AVPlayerItemStatusReadyToPlay then both canPlayFastForward
  // and canPlaySlowForward are always false and it is unknown whether video can
  // be played at these speeds, updatePlayingState will be called again when
  // status changes to AVPlayerItemStatusReadyToPlay.
  float speed = _targetPlaybackSpeed.floatValue;
  BOOL readyToPlay = _player.currentItem.status == AVPlayerItemStatusReadyToPlay;
  if (speed > 2.0 && !_player.currentItem.canPlayFastForward) {
    if (!readyToPlay) {
      return;
    }
    speed = 2.0;
  }
  if (speed < 1.0 && !_player.currentItem.canPlaySlowForward) {
    if (!readyToPlay) {
      return;
    }
    speed = 1.0;
  }
  _player.rate = speed;
}

- (void)sendFailedToLoadVideoEvent {
  // Prefer more detailed error information from tracks loading.
  NSError *error;
  if ([self.player.currentItem.asset statusOfValueForKey:@"tracks"
                                                   error:&error] != AVKeyValueStatusFailed) {
    error = self.player.currentItem.error;
  }
  __block NSMutableOrderedSet<NSString *> *details =
      [NSMutableOrderedSet orderedSetWithObject:@"Failed to load video"];
  void (^add)(NSString *) = ^(NSString *detail) {
    if (detail != nil) {
      [details addObject:detail];
    }
  };
  NSError *underlyingError = error.userInfo[NSUnderlyingErrorKey];
  add(error.localizedDescription);
  add(error.localizedFailureReason);
  add(underlyingError.localizedDescription);
  add(underlyingError.localizedFailureReason);
  NSString *message = [details.array componentsJoinedByString:@": "];
  [self.eventListener videoPlayerDidErrorWithMessage:message];
}

- (void)reportInitialized {
  AVPlayerItem *currentItem = self.player.currentItem;
  NSAssert(currentItem.status == AVPlayerItemStatusReadyToPlay,
           @"reportInitializedIfReadyToPlay was called when the item wasn't ready to play.");
  NSAssert(!_isInitialized, @"reportInitializedIfReadyToPlay should only be called once.");

  _isInitialized = YES;
  [self.eventListener videoPlayerDidInitializeWithDuration:self.duration
                                                      size:currentItem.presentationSize];
}

#pragma mark - FVPVideoPlayerInstanceApi

- (void)playWithError:(FlutterError *_Nullable *_Nonnull)error {
  _isPlaying = YES;
  [self updatePlayingState];
}

- (void)pauseWithError:(FlutterError *_Nullable *_Nonnull)error {
  _isPlaying = NO;
  [self updatePlayingState];
}

- (nullable NSNumber *)position:(FlutterError *_Nullable *_Nonnull)error {
  return @(FVPCMTimeToMillis([_player currentTime]));
}

- (void)seekTo:(NSInteger)position completion:(void (^)(FlutterError *_Nullable))completion {
  CMTime targetCMTime = CMTimeMake(position, 1000);
  CMTimeValue duration = _player.currentItem.asset.duration.value;
  // Without adding tolerance when seeking to duration,
  // seekToTime will never complete, and this call will hang.
  // see issue https://github.com/flutter/flutter/issues/124475.
  CMTime tolerance = position == duration ? CMTimeMake(1, 1000) : kCMTimeZero;
  [_player seekToTime:targetCMTime
        toleranceBefore:tolerance
         toleranceAfter:tolerance
      completionHandler:^(BOOL completed) {
        if (completion) {
          dispatch_async(dispatch_get_main_queue(), ^{
            completion(nil);
          });
        }
      }];
}

- (void)setLooping:(BOOL)looping error:(FlutterError *_Nullable *_Nonnull)error {
  _isLooping = looping;
}

- (void)setVolume:(double)volume error:(FlutterError *_Nullable *_Nonnull)error {
  _player.volume = (float)((volume < 0.0) ? 0.0 : ((volume > 1.0) ? 1.0 : volume));
}

- (void)setPlaybackSpeed:(double)speed error:(FlutterError *_Nullable *_Nonnull)error {
  _targetPlaybackSpeed = @(speed);
  [self updatePlayingState];
}

- (nullable NSArray<FVPMediaSelectionAudioTrackData *> *)getAudioTracks:
    (FlutterError *_Nullable *_Nonnull)error {
  AVPlayerItem *currentItem = _player.currentItem;
  NSAssert(currentItem, @"currentItem should not be nil");
  AVAsset *asset = currentItem.asset;

  // Get tracks from media selection (for HLS streams)
  AVMediaSelectionGroup *audioGroup =
      [asset mediaSelectionGroupForMediaCharacteristic:AVMediaCharacteristicAudible];

  NSMutableArray<FVPMediaSelectionAudioTrackData *> *mediaSelectionTracks =
      [[NSMutableArray alloc] init];

  if (audioGroup.options.count > 0) {
    AVMediaSelection *mediaSelection = currentItem.currentMediaSelection;
    AVMediaSelectionOption *currentSelection =
        [mediaSelection selectedMediaOptionInMediaSelectionGroup:audioGroup];

    for (NSInteger i = 0; i < audioGroup.options.count; i++) {
      AVMediaSelectionOption *option = audioGroup.options[i];
      NSString *displayName = option.displayName;

      NSString *languageCode = nil;
      if (option.locale) {
        languageCode = option.locale.languageCode;
      }

      NSArray<AVMetadataItem *> *titleItems =
          [AVMetadataItem metadataItemsFromArray:option.commonMetadata
                                         withKey:AVMetadataCommonKeyTitle
                                        keySpace:AVMetadataKeySpaceCommon];
      NSString *commonMetadataTitle = titleItems.firstObject.stringValue;

      BOOL isSelected = [currentSelection isEqual:option];

      FVPMediaSelectionAudioTrackData *trackData =
          [FVPMediaSelectionAudioTrackData makeWithIndex:i
                                             displayName:displayName
                                            languageCode:languageCode
                                              isSelected:isSelected
                                     commonMetadataTitle:commonMetadataTitle];

      [mediaSelectionTracks addObject:trackData];
    }
  }

  return mediaSelectionTracks;
}

- (void)selectAudioTrackAtIndex:(NSInteger)trackIndex
                          error:(FlutterError *_Nullable __autoreleasing *_Nonnull)error {
  AVPlayerItem *currentItem = _player.currentItem;
  NSAssert(currentItem, @"currentItem should not be nil");
  AVAsset *asset = currentItem.asset;

  AVMediaSelectionGroup *audioGroup =
      [asset mediaSelectionGroupForMediaCharacteristic:AVMediaCharacteristicAudible];

  if (audioGroup && trackIndex >= 0 && trackIndex < (NSInteger)audioGroup.options.count) {
    AVMediaSelectionOption *option = audioGroup.options[trackIndex];
    [currentItem selectMediaOption:option inMediaSelectionGroup:audioGroup];
  }
}

#pragma mark - Private

- (int64_t)duration {
  // Note: https://openradar.appspot.com/radar?id=4968600712511488
  // `[AVPlayerItem duration]` can be `kCMTimeIndefinite`,
  // use `[[AVPlayerItem asset] duration]` instead.
  return FVPCMTimeToMillis([[[_player currentItem] asset] duration]);
}

@end
