use {
    base::SerialPort,
    std::{
        collections::HashMap,
        io::{Read, Write},
    },
};

#[allow(dead_code)]
#[repr(u8)]
enum Instruction {
    Ping = 0x01,
    Read = 0x02,
    Write = 0x03,
    RegRead = 0x04,
    Action = 0x05,
    Reset = 0x06,
    SyncRead = 0x82,
    SyncWrite = 0x83,
}

#[repr(u8)]
pub enum Register {
    /// firmware version (u16)
    Firmware = 0x00,

    /// model number (u16)
    Model = 0x03,

    /// servo ID (u8)
    Id = 0x05,

    /// baud rate (u8, 0: 1000000, 1: 500000, 2: 250000, 3: 128000, 4: 115200, 5: 76800, 6: 57600, 7: 38400)
    BaudRate = 0x06,

    /// return delay time in 2us steps (u8)
    ReturnDelay = 0x07,

    /// return level (u8, 0: no return, 1: return)
    ReturnLevel = 0x08,

    /// minimum position (u16, 0..4094 in 360/4096 degrees)
    Min = 0x09,

    /// maximum position (u16, 1..4095 in 360/4096 degrees)
    Max = 0x0B,

    /// maximum temperature (u8, 0..100 in deg.C)
    MaxTemperature = 0x0D,

    /// maximum voltage (u8, 0..254 in 0.1V)
    MaxVoltage = 0x0E,

    /// minimum voltage (u8, 0..254 in 0.1V)
    MinVoltage = 0x0F,

    /// maximum torque (u16, 0..1000 in 1/1000 of full torque)
    MaxTorque = 0x10,

    /// phase (u8, unknown)
    Phase = 0x12,

    /// alarm shutdown (u8, bit0:..bit5)
    AlarmShutdown = 0x13,

    /// alarm LED (u8, bit0:..bit5)
    AlarmLed = 0x14,

    /// PID compliance P factor (u8, 0..254)
    ComplianceP = 0x15,

    /// PID compliance D factor (u8, 0..254)
    ComplianceD = 0x16,

    /// PID compliance I factor (u8, 0..254)
    ComplianceI = 0x17,

    /// punch (u16, 0..1000 in 1/1000 of full torque)
    Punch = 0x18,

    /// dead zone clockwise (u8, 0..32 in 360/4096 degrees)
    DeadCw = 0x1A,

    /// dead zone counter-clockwise (u8, 0..32 in 360/4096 degrees)
    DeadCcw = 0x1B,

    /// maximum current (u16, 0..511 in 6.5mA)
    MaxCurrent = 0x1C,

    /// position offset (u16, -2047..2047 in 360/4096 degrees)
    Offset = 0x1F,

    /// mode (u8, 0: position, 1: speed, 2: open loop, 3: step)
    Mode = 0x21,

    /// torque protection limit (u8, 0..100 in 1/100 of full torque)
    ProtectTorque = 0x22,

    /// overtorque time (u8, 0..100 in 10ms)
    OvertorqueTime = 0x23,

    /// overload torque (u8, 0..100 in 1%)
    OverloadTorque = 0x24,

    /// speed compliance P factor (u8, 0..254)
    SpeedComplianceP = 0x25,

    /// overcurrent time (u8, 0..100 in 10ms)
    OvercurrentTime = 0x26,

    /// speed compliance I factor (u8, 0..254)
    SpeedComplianceI = 0x27,

    /// torque enable (u8, 0: disable torque, 1: enable torque, 128: reset current position to 2048)
    TorqueEnable = 0x28,

    /// acceleration (u8, 0..254 in 36000/4096 degrees/s^2)
    Acceleration = 0x29,

    /// goal position (u16, -32766..32766 in 360/4096 degrees)
    GoalPosition = 0x2A,

    /// goal load (u16, 0..1000 in 1/1000 of full torque)
    GoalSpeed = 0x2E,

    /// lock (u8, 0: lock EPROM, 1: unlock EPROM)
    Lock = 0x37,

    /// position (u16, 0..4095 in 360/4096 degrees)
    Position = 0x38,

    /// speed (u16, 0..4095 in 360/4096 degrees/s)
    Speed = 0x3A,

    /// load (u16, 0..1000 in 1/1000 of full torque)
    Load = 0x3C,

    /// voltage (u8, 0..254 in 0.1V)
    Voltage = 0x3E,

    /// temperature (u8, 0..100 in deg.C)
    Temperature = 0x3F,

    /// async write (u8, unknown)
    AsyncWrite = 0x40,

    /// status (u8, bit0: voltage, bit1: sensor, bit2: temperature, bit3: current, bit4: position, bit5: overload)
    Status = 0x41,

    /// mobile sign (u8, unknown)
    MobileSign = 0x42,

    /// virtual position (u16, unknown)
    VirPosition = 0x43,

    /// current (u16, 0..500 in 6.5mA)
    Current = 0x45,
}

pub const RADIANS_PER_TICK: f32 = std::f32::consts::TAU / 4096.0;

pub fn encode_position(position: f32) -> u16 {
    let value = (position / RADIANS_PER_TICK) as i16 + 2048;
    if value < 0 {
        return 0x8000 | (-value as u16);
    } else {
        return value as u16;
    }
}

pub fn decode_position(value: u16) -> f32 {
    if (value & 0x8000) != 0 {
        return ((-((value & 0x7FFF) as i16) - 2048) as f32) * RADIANS_PER_TICK;
    } else {
        return (((value as i16) - 2048) as f32) * RADIANS_PER_TICK;
    }
}

pub fn encode_offset(offset: f32) -> u16 {
    let value = (offset / RADIANS_PER_TICK) as i16;
    if value < 0 {
        return 0x0800 | (-value as u16);
    } else {
        return value as u16;
    }
}

pub fn decode_offset(value: u16) -> f32 {
    if (value & 0x0800) != 0 {
        return (-((value & 0x07FF) as i16) as f32) * RADIANS_PER_TICK;
    } else {
        return (value as f32) * RADIANS_PER_TICK;
    }
}

pub fn encode_speed(speed: f32) -> u16 {
    let value = (speed / RADIANS_PER_TICK) as i16;
    if value < 0 {
        return 0x8000 | (-value as u16);
    } else {
        return value as u16;
    }
}

pub fn decode_speed(value: u16) -> f32 {
    if (value & 0x8000) != 0 {
        return ((-((value & 0x7FFF) as i16)) as f32) * RADIANS_PER_TICK;
    } else {
        return (value as f32) * RADIANS_PER_TICK;
    }
}

pub fn encode_acceleration(acceleration: f32) -> u8 {
    return (acceleration / (100.0 * RADIANS_PER_TICK)) as u8;
}

pub fn encode_load(load: f32) -> u16 {
    let value = (load * 1000.0) as i16;
    if value < 0 {
        return 0x0400 | ((-value & 0x03FF) as u16);
    } else {
        return value as u16;
    }
}

pub fn decode_load(value: u16) -> f32 {
    if (value & 0x0400) != 0 {
        return (-((value & 0x03FF) as i16) as f32) * 0.001;
    } else {
        return (value as f32) * 0.001;
    }
}

pub fn encode_voltage(voltage: f32) -> u8 {
    return (voltage * 10.0) as u8;
}

pub fn decode_voltage(value: u8) -> f32 {
    return (value as f32) * 0.1;
}

pub fn encode_temperature(temperature: f32) -> u8 {
    return temperature as u8;
}

pub fn decode_temperature(value: u8) -> f32 {
    return value as f32;
}

pub fn decode_current(value: u16) -> f32 {
    return (value as f32) * 0.0065;
}

fn calculate_checksum(data: &mut [u8]) {
    let sum: u8 = !data[2..data.len() - 1]
        .iter()
        .fold(0, |sum, &b| sum.wrapping_add(b));
    data[data.len() - 1] = sum;
}

fn verify_checksum(data: &[u8]) -> bool {
    let sum: u8 = !data[2..data.len() - 1]
        .iter()
        .fold(0, |sum, &b| sum.wrapping_add(b));
    data[data.len() - 1] == sum
}

#[derive(Clone)]
pub enum ServoMode {
    Position,
    Speed,
    Load,
}

pub struct Servo {
    mode: ServoMode,
    min_voltage: f32,
    max_voltage: f32,
    max_temperature: f32,
    offset: f32,
}

pub struct Bus {
    port: SerialPort,
    servos: HashMap<usize, Servo>,
}

impl Bus {
    pub fn new(port: SerialPort) -> Result<Self, std::io::Error> {
        Ok(Bus {
            port,
            servos: HashMap::new(),
        })
    }

    /// Write a packet to the bus and wait for TX to complete.
    fn write_packet(&mut self, packet: &[u8]) -> Result<(), std::io::Error> {
        self.port.flush_input()?;
        let n = self.port.write(packet)?;
        if n != packet.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!(
                    "Feetech: write failed (expected {} bytes, got {})",
                    packet.len(),
                    n
                ),
            ));
        }
        // wait for all bytes to leave the UART before switching to RX
        self.port.flush()?;
        Ok(())
    }

    pub fn send_ping(&mut self, id: usize) -> Result<(), std::io::Error> {
        let mut request = [0xFF, 0xFF, id as u8, 2, Instruction::Ping as u8, 0];
        calculate_checksum(&mut request);
        self.write_packet(&request)
    }

    pub fn recv_ping(&mut self) -> Result<usize, std::io::Error> {
        let mut response = [0u8; 6];
        match self.port.read(&mut response) {
            Ok(bytes_read) => {
                if bytes_read != response.len() {
                    Err(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!(
                            "Feetech: ping read failed (expected {} bytes, got {})",
                            response.len(),
                            bytes_read
                        ),
                    ))
                } else if verify_checksum(&response) {
                    Ok(response[2] as usize)
                } else {
                    Err(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        "Feetech: ping checksum error",
                    ))
                }
            }
            Err(error) => Err(error),
        }
    }

    pub fn send_ping_all(&mut self) -> Result<(), std::io::Error> {
        let mut request = [0xFF, 0xFF, 0xFE, 2, Instruction::Ping as u8, 0];
        calculate_checksum(&mut request);
        self.write_packet(&request)
    }

    pub fn recv_ping_all(&mut self) -> Result<Vec<usize>, std::io::Error> {
        let mut responses = [0u8; 6 * 32];
        match self.port.read(&mut responses) {
            Ok(bytes_read) => {
                if bytes_read >= 6 {
                    let mut ids = Vec::<usize>::new();
                    let mut index = 0;
                    while index <= bytes_read - 6 {
                        if (responses[index] == 0xFF) && (responses[index + 1] == 0xFF) {
                            let response = &responses[index..index + 6];
                            if verify_checksum(response) {
                                ids.push(response[2] as usize);
                            }
                            index += 6;
                        } else {
                            index += 1;
                        }
                    }
                    Ok(ids)
                } else {
                    Ok(Vec::new())
                }
            }
            Err(error) => Err(error),
        }
    }

    pub fn send_write(
        &mut self,
        id: usize,
        start: Register,
        data: &[u8],
    ) -> Result<(), std::io::Error> {
        let mut request = vec![
            0xFF,
            0xFF,
            id as u8,
            3 + data.len() as u8,
            Instruction::Write as u8,
            start as u8,
        ];
        request.extend(data);
        request.push(0);
        calculate_checksum(&mut request);
        self.write_packet(&request)
    }

    pub fn send_read(
        &mut self,
        id: usize,
        start: Register,
        length: u8,
    ) -> Result<(), std::io::Error> {
        let mut request = [
            0xFF,
            0xFF,
            id as u8,
            4,
            Instruction::Read as u8,
            start as u8,
            length,
            0,
        ];
        calculate_checksum(&mut request);
        self.write_packet(&request)
    }

    pub fn recv_read(&mut self, length: usize) -> Result<Vec<u8>, std::io::Error> {
        let mut response = vec![0u8; 6 + length];
        match self.port.read(&mut response) {
            Ok(bytes_read) => {
                if bytes_read != response.len() {
                    Err(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!(
                            "Feetech: read failed (expected {} bytes, got {})",
                            response.len(),
                            bytes_read
                        ),
                    ))
                } else {
                    if (response[0] != 0xFF) || (response[1] != 0xFF) {
                        Err(std::io::Error::new(
                            std::io::ErrorKind::Other,
                            "Feetech: read failed",
                        ))
                    } else {
                        if !verify_checksum(&response) {
                            Err(std::io::Error::new(
                                std::io::ErrorKind::Other,
                                "Feetech: read checksum error",
                            ))
                        } else {
                            Ok(response[5..5 + length].to_vec())
                        }
                    }
                }
            }
            _ => Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Feetech: read failed",
            )),
        }
    }

    pub fn send_sync_write(
        &mut self,
        start: Register,
        data: HashMap<usize, Vec<u8>>,
    ) -> Result<(), std::io::Error> {
        let length = data.values().next().unwrap().len();
        let count = data.len();
        let mut request = vec![
            0xFF,
            0xFF,
            0xFE,
            4 + ((1 + length) * count) as u8,
            Instruction::SyncWrite as u8,
            start as u8,
            length as u8,
        ];
        for (&id, buffer) in data.iter() {
            request.push(id as u8);
            request.extend(buffer);
        }
        request.push(0);
        calculate_checksum(&mut request);
        self.write_packet(&request)
    }

    pub fn send_sync_read(
        &mut self,
        start: Register,
        length: u8,
        ids: &[usize],
    ) -> Result<(), std::io::Error> {
        let mut request = vec![
            0xFF,
            0xFF,
            0xFE,
            4 + ids.len() as u8,
            Instruction::SyncRead as u8,
            start as u8,
            length,
        ];
        request.extend(ids.iter().map(|&id| id as u8));
        request.push(0);
        calculate_checksum(&mut request);
        self.write_packet(&request)
    }

    pub async fn recv_sync_read(
        &mut self,
        count: usize,
        length: usize,
    ) -> Result<HashMap<u8, Vec<u8>>, std::io::Error> {
        let mut responses = vec![0u8; (6 + length) * count];
        let bytes_read = match self.port.read(&mut responses) {
            Ok(bytes_read) => bytes_read,
            _ => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "Feetech: sync read failed",
                ))
            }
        };
        let mut index = 0;
        let mut data = HashMap::<u8, Vec<u8>>::new();
        if bytes_read >= 6 + length {
            while index <= bytes_read - 6 - length {
                if (responses[index] == 0xFF) && (responses[index + 1] == 0xFF) {
                    let response = &responses[index..index + 6 + length];
                    if verify_checksum(&response) {
                        data.insert(response[2], response[5..5 + length].to_vec());
                    }
                    index += 6 + length;
                } else {
                    index += 1;
                }
            }
        }
        Ok(data)
    }

    fn init_servo(&mut self, id: usize) -> Result<(), std::io::Error> {
        // unlock servo
        self.send_write(id, Register::Lock, &[1])?;

        // write parameter registers
        let data = {
            let servo = self.servos.get(&id).unwrap();
            let min_voltage = encode_voltage(servo.min_voltage);
            let max_voltage = encode_voltage(servo.max_voltage);
            let max_temperature = encode_temperature(servo.max_temperature);
            [
                0,
                0,
                0,
                0,
                0xFF,
                0x0F,
                max_temperature,
                max_voltage,
                min_voltage,
            ]
        };
        self.send_write(id, Register::ReturnDelay, &data)?;

        // write offset and mode register
        let data = {
            let servo = self.servos.get(&id).unwrap();
            let offset = encode_offset(servo.offset);
            let mode = match servo.mode {
                ServoMode::Position => 0,
                ServoMode::Speed => 1,
                ServoMode::Load => 2,
            };
            [(offset & 0xFF) as u8, (offset >> 8) as u8, mode]
        };
        self.send_write(id, Register::Offset, &data)?;

        // lock servo
        self.send_write(id, Register::Lock, &[0])?;

        Ok(())
    }

    pub fn add_servo(
        &mut self,
        id: usize,
        min_voltage: f32,
        max_voltage: f32,
        max_temperature: f32,
        offset: f32,
    ) -> Result<(), std::io::Error> {
        self.servos.insert(
            id,
            Servo {
                mode: ServoMode::Position,
                min_voltage,
                max_voltage,
                max_temperature,
                offset,
            },
        );
        self.init_servo(id)?;
        Ok(())
    }

    pub fn remove_servo(&mut self, id: usize) {
        self.servos.remove(&id);
    }

    pub fn reset_servo(&mut self, id: usize) -> Result<f32, std::io::Error> {
        if self.servos.contains_key(&id) {
            self.send_write(id, Register::TorqueEnable, &[128])?;
            std::thread::sleep(std::time::Duration::from_millis(500));
            self.send_read(id, Register::Offset, 2)?;
            std::thread::sleep(std::time::Duration::from_millis(100));
            let response = self.recv_read(2)?;
            let offset = decode_offset((response[1] as u16) << 8 | response[0] as u16);
            Ok(offset)
        } else {
            Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Feetech: servo not found",
            ))
        }
    }
}
