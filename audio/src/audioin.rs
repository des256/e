use {
    base::{channel, Receiver, RecvError, Sender, TryRecvError},
    libpulse_binding::{
        def::BufferAttr,
        sample::{Format, Spec},
        stream::Direction,
    },
    libpulse_simple_binding::Simple,
    std::time::Duration,
};

/// Configuration for audio capture.
pub struct AudioInConfig {
    /// PulseAudio source device name, or `None` for the default source.
    pub device_name: Option<String>,
    /// Number of audio channels (e.g. 1 for mono, 2 for stereo).
    pub channels: usize,
    /// Sample rate in Hz.
    pub sample_rate: usize,
    /// Number of frames per chunk delivered to the [`Listener`].
    ///
    /// Each chunk contains `chunk_size * channels` interleaved `f32` samples.
    pub chunk_size: usize,
}

impl Default for AudioInConfig {
    fn default() -> Self {
        Self {
            device_name: None,
            channels: 1,
            sample_rate: 16000,
            chunk_size: 1024,
        }
    }
}

/// Control handle for a running audio capture thread.
///
/// Dropping the handle closes the config channel but does **not** stop the
/// capture thread — it runs until the [`Listener`] is also dropped.
pub struct Handle {
    config_tx: Sender<AudioInConfig>,
}

/// Receives captured audio chunks from the background thread.
///
/// Each chunk is a `Vec<f32>` of `chunk_size * channels` interleaved samples.
/// Dropping the listener disconnects the audio channel and stops the capture
/// thread.
pub struct Listener {
    audio_rx: Receiver<Vec<f32>>,
}

/// Spawn an audio capture thread with the given initial configuration.
///
/// Returns a [`Handle`] for reconfiguration and a [`Listener`] for receiving
/// captured audio. The thread opens a PulseAudio recording stream and
/// continuously delivers chunks of `f32` PCM data. If the PulseAudio
/// connection fails, the thread retries after a 1-second delay.
pub fn create(config: AudioInConfig) -> (Handle, Listener) {
    let (audio_tx, audio_rx) = channel::<Vec<f32>>();
    let (config_tx, config_rx) = channel::<AudioInConfig>();
    std::thread::spawn({
        move || {
            let mut config = config;
            let mut pulse: Option<Simple> = None;
            let mut buffer: Vec<f32> = Vec::new();
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
                    let bytes_per_chunk = samples_per_chunk * 4;
                    buffer = vec![0.0f32; samples_per_chunk];
                    let buffer_attr = BufferAttr {
                        maxlength: bytes_per_chunk as u32 * 16,
                        tlength: u32::MAX,
                        prebuf: u32::MAX,
                        minreq: u32::MAX,
                        fragsize: bytes_per_chunk as u32,
                    };
                    pulse = match Simple::new(
                        None,
                        "e-audioin",
                        Direction::Record,
                        config.device_name.as_deref(),
                        "audio-capture",
                        &spec,
                        None,
                        Some(&buffer_attr),
                    ) {
                        Ok(pulse) => Some(pulse),
                        Err(error) => {
                            base::error!("AudioIn: failed to open pulse: {}", error);
                            std::thread::sleep(Duration::from_secs(1));
                            None
                        }
                    };
                }

                // if pulse is open, process a chunk
                if let Some(inner_pulse) = &pulse {
                    // reinterpret f32 buffer as u8 slice for PulseAudio read
                    let byte_slice = unsafe {
                        std::slice::from_raw_parts_mut(
                            buffer.as_mut_ptr() as *mut u8,
                            buffer.len() * 4,
                        )
                    };
                    match inner_pulse.read(byte_slice) {
                        Ok(()) => {
                            // send to listener
                            if let Err(_) = audio_tx.send(buffer.clone()) {
                                // if send fails, break the entire thread
                                base::error!("AudioIn: audio channel disconnected");
                                return;
                            }
                        }
                        Err(error) => {
                            // reading failed, so close pulse and wait for 1 second before trying again
                            base::error!("AudioIn: failed to read from PulseAudio: {}", error);
                            pulse = None;
                            std::thread::sleep(Duration::from_secs(1));
                        }
                    }
                }
            }
        }
    });
    (Handle { config_tx }, Listener { audio_rx })
}

impl Handle {
    /// Send a new configuration to the capture thread.
    ///
    /// The thread picks up the config on its next iteration, closes the
    /// current PulseAudio stream, and reopens with the new parameters.
    pub fn configure(&self, config: AudioInConfig) {
        if let Err(_) = self.config_tx.send(config) {
            base::error!("AudioIn: failed to send config");
        }
    }
}

impl Listener {
    /// Wait for the next audio chunk.
    ///
    /// Returns `Err(RecvError::Disconnected)` if the capture thread has exited.
    pub async fn recv(&self) -> Result<Vec<f32>, RecvError> {
        match self.audio_rx.recv().await {
            Some(audio) => Ok(audio),
            None => Err(RecvError::Disconnected),
        }
    }

    /// Non-blocking receive. Returns `None` if no chunk is available yet
    /// or the channel has disconnected.
    pub fn try_recv(&self) -> Option<Vec<f32>> {
        match self.audio_rx.try_recv() {
            Ok(audio) => Some(audio),
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => {
                base::error!("AudioIn: audio channel disconnected");
                None
            }
        }
    }
}
