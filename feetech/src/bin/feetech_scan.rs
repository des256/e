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
    // all Feetech baud rates (register values 0..7)
    for baud_rate in [1000000, 500000, 250000, 128000, 115200, 76800, 57600, 38400] {
        println!("Trying baud rate: {}", baud_rate);
        let port = base::SerialPort::open(path, baud_rate)?;
        let mut bus = feetech::Bus::new(port, libc::TIOCM_DTR)?;
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
    Ok(())
}
