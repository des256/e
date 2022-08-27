pub fn compile_vertex_shader(items: Vec<sr::Item>,_vertex: Vec<(String,sr::BaseType)>) -> Option<Vec<u8>> {
    println!("VERTEX SHADER:\ninput:");
    for item in items {
        println!("{}",item);
    }

    let _r: Vec<u32> = vec![0];

    Some(vec![0])
}

pub fn compile_fragment_shader(items: Vec<sr::Item>) -> Option<Vec<u8>> {
    println!("FRAGMENT SHADER:\ninput:");
    for item in items {
        println!("{}",item);
    }

    let _r: Vec<u32> = vec![0];

    Some(vec![0])
}
