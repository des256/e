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

pub struct AudioInConfig {
    pub device_name: Option<String>,
    pub channels: usize,
    pub sample_rate: usize,
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

pub struct Handle {
    config_tx: Sender<AudioInConfig>,
}

pub struct Listener {
    input_rx: Receiver<Vec<f32>>,
}

pub fn create(config: AudioInConfig) -> (Handle, Listener) {
    let (input_tx, input_rx) = channel::<Vec<f32>>();
    let (config_tx, config_rx) = channel::<AudioInConfig>();
    std::thread::spawn({
        move || {
            let mut config = config;
            let mut pulse: Option<Simple> = None;
            let mut buffer: Vec<u8> = Vec::new();
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
                    let bytes_per_chunk = config.chunk_size * config.channels * 4;
                    buffer = vec![0u8; bytes_per_chunk];
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
                        match config.device_name {
                            Some(ref name) => Some(name.as_str()),
                            None => None,
                        },
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
                    match inner_pulse.read(&mut buffer) {
                        Ok(()) => {
                            // convert buffer to final format
                            let buffer = unsafe {
                                std::slice::from_raw_parts(
                                    buffer.as_ptr() as *const f32,
                                    config.chunk_size * config.channels,
                                )
                                .to_vec()
                            };

                            // send to listener
                            if let Err(_) = input_tx.send(buffer) {
                                // if send fails, break the entire thread
                                base::error!("AudioIn: failed to send audio");
                                break;
                            }
                        }
                        Err(error) => {
                            // reading failed, so close pulse and wait for 1 second before trying again
                            base::error!("AudioIn: failed to read pulse: {}", error);
                            pulse = None;
                            std::thread::sleep(Duration::from_secs(1));
                        }
                    }
                }
            }
        }
    });
    (Handle { config_tx }, Listener { input_rx })
}

impl Handle {
    pub fn configure(&self, config: AudioInConfig) {
        if let Err(_) = self.config_tx.send(config) {
            base::error!("AudioIn: failed to send config");
        }
    }
}

impl Listener {
    pub async fn recv(&self) -> Result<Vec<f32>, RecvError> {
        match self.input_rx.recv().await {
            Some(sample) => Ok(sample),
            None => Err(RecvError::Disconnected),
        }
    }

    pub fn try_recv(&self) -> Option<Vec<f32>> {
        match self.input_rx.try_recv() {
            Ok(sample) => Some(sample),
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => {
                base::error!("AudioIn: data channel disconnected");
                None
            }
        }
    }
}
