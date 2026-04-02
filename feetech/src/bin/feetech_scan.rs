use {feetech::feetech, serialport::available_ports, std::time::Duration};

fn main() -> Result<(), std::io::Error> {
    let ports = available_ports()?;
    for (index, port) in ports.iter().enumerate() {
        println!("{}: {}", index, port.port_name);
    }
    println!("Enter the index of the port to use: ");
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    let index = input.trim().parse::<usize>().unwrap();
    let path = &ports[index].port_name;
    for baud_rate in [1000000, 500000, 250000, 128000, 115200, 76800, 57600, 38400] {
        println!("Trying baud rate: {}", baud_rate);
        let port = serialport::new(path, baud_rate).open()?;
        let mut bus = feetech::Bus::new(port)?;
        bus.send_ping_all()?;
        std::thread::sleep(Duration::from_secs(1));
        let ids = bus.recv_ping_all()?;
        if ids.len() > 0 {
            println!("Found {} servos:", ids.len());
            for id in ids {
                println!("  ID: {}", id);
            }
        }
    }
    Ok(())
}
