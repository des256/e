use {base::available_ports, feetech::feetech, std::time::Duration};

fn main() -> Result<(), std::io::Error> {
    let ports = available_ports();
    for (index, port) in ports.iter().enumerate() {
        println!("{}: {}", index, port);
    }
    println!("Enter the index of the port to use: ");
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    let index = input.trim().parse::<usize>().unwrap();
    let path = &ports[index].path;
    for baud_rate in [1000000, 500000, 115200, 57600, 38400] {
        for rts_on_send in [true, false] {
            println!(
                "Trying baud rate: {}, RS-485 rts_on_send: {}",
                baud_rate, rts_on_send
            );
            let port = base::SerialPort::open(path, baud_rate)?;
            match port.enable_rs485(rts_on_send) {
                Ok(()) => println!("  RS-485 mode enabled"),
                Err(e) => println!("  RS-485 mode not supported: {}", e),
            }
            let mut bus = feetech::Bus::new(port)?;
            bus.send_ping_all()?;
            std::thread::sleep(Duration::from_secs(1));
            let ids = bus.recv_ping_all()?;
            if !ids.is_empty() {
                println!("Found {} servos:", ids.len());
                for id in ids {
                    println!("  ID: {}", id);
                }
            }
        }
    }
    Ok(())
}
