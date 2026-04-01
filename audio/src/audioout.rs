use {
    base::{channel, Epoch, Receiver, RecvError, Sender, TryRecvError},
    libpulse_binding::{
        sample::{Format, Spec},
        stream::Direction,
    },
    libpulse_simple_binding::Simple,
    std::{sync::Arc, time::Duration},
};

/// Configuration for audio playback.
pub struct AudioOutConfig {
    /// PulseAudio sink device name, or `None` for the default sink.
    pub device_name: Option<String>,
    /// Number of audio channels (e.g. 1 for mono, 2 for stereo).
    pub channels: usize,
    /// Sample rate in Hz.
    pub sample_rate: usize,
    /// Number of frames per chunk written to PulseAudio.
    ///
    /// Each chunk contains `chunk_size * channels` interleaved `f32` samples.
    pub chunk_size: usize,
}

impl Default for AudioOutConfig {
    fn default() -> Self {
        Self {
            device_name: None,
            channels: 1,
            sample_rate: 24000,
            chunk_size: 1024,
        }
    }
}

/// A chunk of audio to be played, carrying a user-defined payload `T`.
pub struct Audio<T: Clone + Send + 'static> {
    /// User-defined metadata attached to this audio (e.g. an identifier).
    /// Cloned into [`Status`] notifications.
    pub payload: T,
    /// Interleaved `f32` PCM samples.
    pub data: Vec<f32>,
    /// Epoch stamp for staleness checking. Audio whose stamp does not match
    /// the current [`Epoch`] value is canceled with a [`Status::Canceled`]
    /// notification.
    pub stamp: u64,
}

/// Playback status notification sent to the [`Listener`].
pub enum Status<T: Clone + Send + 'static> {
    /// Playback of an audio chunk has begun.
    Started {
        /// Payload from the [`Audio`] that started playing.
        payload: T,
    },
    /// An audio chunk finished playing to completion.
    Finished {
        /// Payload from the completed [`Audio`].
        payload: T,
        /// Sample index at which playback ended (equals `data.len()`).
        index: usize,
    },
    /// An audio chunk was canceled before finishing.
    Canceled {
        /// Payload from the canceled [`Audio`].
        payload: T,
        /// Sample index at which playback was interrupted.
        index: usize,
    },
}

/// Control handle for a running audio playback thread.
///
/// Dropping the handle closes both the audio and config channels, which
/// causes the playback thread to exit.
pub struct Handle<T: Clone + Send + 'static> {
    audio_tx: Sender<Audio<T>>,
    config_tx: Sender<AudioOutConfig>,
}

/// Receives playback [`Status`] notifications from the background thread.
pub struct Listener<T: Clone + Send + 'static> {
    status_rx: Receiver<Status<T>>,
}

/// Spawn an audio playback thread with the given initial configuration.
///
/// The `epoch` is used for staleness checking: audio whose
/// [`stamp`](Audio::stamp) does not match [`Epoch::current`] is skipped.
/// This allows bulk-canceling queued audio by advancing the epoch.
///
/// Returns a [`Handle`] for submitting audio and reconfiguring, and a
/// [`Listener`] for receiving playback status notifications. The thread
/// continuously writes to PulseAudio, outputting silence when no audio is
/// queued. If the PulseAudio connection fails, the thread retries after a
/// 1-second delay.
pub fn create<T: Clone + Send + 'static>(
    config: AudioOutConfig,
    epoch: &Arc<Epoch>,
) -> (Handle<T>, Listener<T>) {
    let (audio_tx, audio_rx) = channel::<Audio<T>>();
    let (status_tx, status_rx) = channel::<Status<T>>();
    let (config_tx, config_rx) = channel::<AudioOutConfig>();
    std::thread::spawn({
        let epoch = Arc::clone(&epoch);
        move || {
            let mut config = config;
            let mut pulse: Option<Simple> = None;
            let mut buffer: Vec<f32> = Vec::new();
            let mut current_audio: Option<Audio<T>> = None;
            let mut current_index = 0usize;
            loop {
                // if a new config is waiting, pick it up and close the current pulse, if any
                if let Ok(new_config) = config_rx.try_recv() {
                    config = new_config;
                    pulse = None;
                }

                // if no pulse, open one from the current config
                if pulse.is_none() {
                    let spec = Spec {
                        format: Format::F32le,
                        channels: config.channels as u8,
                        rate: config.sample_rate as u32,
                    };
                    let samples_per_chunk = config.chunk_size * config.channels;
                    buffer = vec![0.0f32; samples_per_chunk];
                    pulse = match Simple::new(
                        None,
                        "e-audioout",
                        Direction::Playback,
                        config.device_name.as_deref(),
                        "audio-playback",
                        &spec,
                        None,
                        None,
                    ) {
                        Ok(pulse) => Some(pulse),
                        Err(error) => {
                            base::error!("AudioOut: failed to open pulse: {}", error);
                            std::thread::sleep(Duration::from_secs(1));
                            None
                        }
                    };
                }

                // if pulse is open, process current chunk
                if let Some(inner_pulse) = &pulse {
                    // build buffer
                    let mut i = 0usize;
                    let samples_per_chunk = config.chunk_size * config.channels;
                    while i < samples_per_chunk {
                        // if currently playing audio, copy max into the buffer
                        if let Some(audio) = &current_audio {
                            // if the current audio is stale, cancel it
                            if !epoch.is_current(audio.stamp) {
                                if let Err(_) = status_tx.send(Status::Canceled {
                                    payload: audio.payload.clone(),
                                    index: current_index,
                                }) {
                                    base::error!("AudioOut: failed to send canceled status");
                                    // TODO: maybe exit thread because channel is broken?
                                }
                                current_audio = None;
                                current_index = 0;
                                continue;
                            }
                            let mut n = audio.data.len() - current_index;
                            if n > samples_per_chunk - i {
                                n = samples_per_chunk - i;
                            }
                            buffer[i..i + n]
                                .copy_from_slice(&audio.data[current_index..current_index + n]);
                            current_index += n;
                            i += n;

                            // if the buffer is now full, send finished status and end this chunk
                            if current_index >= audio.data.len() {
                                // notify about finishing this audio
                                if let Err(_) = status_tx.send(Status::Finished {
                                    payload: audio.payload.clone(),
                                    index: current_index,
                                }) {
                                    base::error!("AudioOut: failed to send finished status");
                                    // TODO: maybe exit thread because channel is broken?
                                }
                                current_audio = None;
                                current_index = 0;
                            }
                        }
                        // if not playing audio, get next one from the channel
                        else {
                            match audio_rx.try_recv() {
                                // new audio from channel
                                Ok(audio) => {
                                    // skip if stale
                                    if !epoch.is_current(audio.stamp) {
                                        if let Err(_) = status_tx.send(Status::Canceled {
                                            payload: audio.payload.clone(),
                                            index: 0,
                                        }) {
                                            base::error!(
                                                "AudioOut: failed to send canceled status"
                                            );
                                            // TODO: maybe exit thread because channel is broken?
                                        }
                                        continue;
                                    }
                                    // notify about starting this audio
                                    if let Err(_) = status_tx.send(Status::Started {
                                        payload: audio.payload.clone(),
                                    }) {
                                        base::error!("AudioOut: failed to send started status");
                                        // TODO: maybe exit thread because channel is broken?
                                    }
                                    current_audio = Some(audio);
                                    current_index = 0;
                                }
                                // channel is empty, so fill buffer with zeros
                                Err(TryRecvError::Empty) => {
                                    if i < samples_per_chunk {
                                        buffer[i..].fill(0.0);
                                        i = samples_per_chunk;
                                    }
                                }
                                // channel is disconnected, so exit thread
                                Err(TryRecvError::Disconnected) => {
                                    base::error!("AudioOut: audio channel disconnected");
                                    return;
                                }
                            }
                        }
                    }

                    // write to pulse
                    let slice = unsafe {
                        std::slice::from_raw_parts(buffer.as_ptr() as *const u8, buffer.len() * 4)
                    };
                    if let Err(error) = inner_pulse.write(&slice) {
                        // writing failed, so close pulse and wait for 1 second before trying again
                        base::error!("AudioOut: failed to write to PulseAudio: {}", error);
                        pulse = None;
                        std::thread::sleep(Duration::from_secs(1));
                    }
                }
            }
        }
    });
    (
        Handle {
            audio_tx,
            config_tx,
        },
        Listener { status_rx },
    )
}

impl<T: Clone + Send + 'static> Handle<T> {
    /// Send a new configuration to the playback thread.
    ///
    /// The thread picks up the config on its next iteration, closes the
    /// current PulseAudio stream, and reopens with the new parameters.
    pub fn configure(&self, config: AudioOutConfig) {
        if let Err(_) = self.config_tx.send(config) {
            base::error!("AudioOut: failed to send config");
        }
    }

    /// Queue an audio chunk for playback.
    ///
    /// The chunk is played in FIFO order after any previously queued audio.
    /// If the audio's [`stamp`](Audio::stamp) is stale when the thread
    /// reaches it, a [`Status::Canceled`] notification is sent instead.
    pub fn send(&self, audio: Audio<T>) {
        if let Err(_) = self.audio_tx.send(audio) {
            base::error!("AudioOut: failed to send audio");
        }
    }
}

impl<T: Clone + Send + 'static> Listener<T> {
    /// Wait for the next playback status notification.
    ///
    /// Returns `Err(RecvError::Disconnected)` if the playback thread has exited.
    pub async fn recv(&self) -> Result<Status<T>, RecvError> {
        match self.status_rx.recv().await {
            Some(status) => Ok(status),
            None => Err(RecvError::Disconnected),
        }
    }

    /// Non-blocking receive. Returns `None` if no status is available yet
    /// or the channel has disconnected.
    pub fn try_recv(&self) -> Option<Status<T>> {
        match self.status_rx.try_recv() {
            Ok(status) => Some(status),
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => {
                base::error!("AudioOut: status channel disconnected");
                None
            }
        }
    }
}
