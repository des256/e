extern crate freetype;

use {
    e::*,
    std::{
        env,
        path::Path,
        fs::File,
        io::prelude::*,
    },
};

static CHARACTERS: &[(u32,u32)] = &[
    (0x0020,0x0080),
    (0x00A0,0x0100),
    (0x0100,0x0180),
    (0x0180,0x0250),

    //(0x0250,0x02B0),  // IPA Extensions
    //(0x02B0,0x0300),  // Spacing Modifier Letters
    //(0x0300,0x0370),  // Combining Diacritical Marks
    
    //(0x0370,0x0400),  // Greek and Coptic
    //(0x0400,0x0500),  // Cyrillic
    //(0x0500,0x0530),  // Cyrillic Supplement
    
    //(0x0530,0x0590),  // Armenian
    //(0x0590,0x0600),  // Hebrew
    //(0x0600,0x0700),  // Arabic
    //(0x0700,0x0750),  // Syriac
    //(0x0750,0x0780),  // Arabic Supplement
    //(0x3040,0x30A0),  // Hiragana
    //(0x30A0,0x3100),  // Katakana
];

struct ImageCharacter {
    size: u32,
    n: u32,
    image: Mat<u8>,
    bearing: Vec2<i32>,
    advance: i32,
}

struct ImageCharacterSet {
    size: u32,
    min: i32,
    max: i32,
}

struct Character {
    n: u32,
    r: Rect<i32>,
    bearing: Vec2<i32>,
    advance: i32,
}

struct CharacterSet {
    size: u32,
    min: i32,
    max: i32,
    characters: Vec<Character>,
}

#[derive(Clone,Copy)]
struct Mapped {
    used: bool,
    value: u8,
}

impl Zero for Mapped { const ZERO: Mapped = Mapped { used: false, value: 0, }; }

fn exit_help() {
    println!("Make Font Texture from TTF File");
    println!();
    println!("USAGE:");
    println!("    mkfnt <infile> <outfile> [FLAGS] [OPTIONS]");
    println!();
    println!("FLAGS:");
    println!("    -h, --help     Prints help information");
    println!("    -v, --version  Prints version information");
    println!();
    println!("OPTIONS:");
    println!("    -s <size>, --size <size>  Font size (default 28), multiple sizes is possible");
    println!("    -d <dim>,  --dim <dim>    Resulting texture dimension (square)");
    std::process::exit(-1);
}

fn exit_version() {
    println!("Make Font Texture from TTF File");
    std::process::exit(-1);
}

fn find_empty(size: &Vec2<usize>,haystack: &Mat<Mapped>,p: &mut Vec2<isize>) -> bool {
    for hy in 0..haystack.size.y - size.y {
        for hx in 0..haystack.size.x - size.x {
            let mut found = true;
            for y in 0..size.y {
                for x in 0..size.x {
                    if haystack[(hx + x,hy + y)].used {
                        found = false;
                        break;
                    }
                }
                if !found {
                    break;
                }
            }
            if found {
                *p = Vec2::<isize> { x: hx as isize,y: hy as isize, };
                return true;
            }
        }
    }
    false
}

fn find_rect(needle: &Mat<u8>,haystack: &Mat<Mapped>,p: &mut Vec2<isize>) -> bool {
    for hy in 0..haystack.size.y - needle.size.y {
        for hx in 0..haystack.size.x - needle.size.x {
            let mut found = true;
            for y in 0..needle.size.y {
                for x in 0..needle.size.x {
                    if haystack[(hx + x,hy + y)].value != needle[(x,y)] {
                        found = false;
                        break;
                    }
                }
                if !found {
                    break;
                }
            }
            if found {
                *p = Vec2::<isize> { x: hx as isize,y: hy as isize, };
                return true;
            }
        }
    }
    false
}

fn push_u32(buf: &mut Vec<u8>,v: u32) {
    buf.push((v & 255) as u8);
    buf.push(((v >> 8) & 255) as u8);
    buf.push(((v >> 16) & 255) as u8);
    buf.push((v >> 24) as u8);
}

fn push_i32(buf: &mut Vec<u8>,v: i32) {
    push_u32(buf,v as u32);
}

fn main() {
    // parse arguments
    let mut args = env::args();
    let _command = args.next().unwrap();
    let infile_wrap = args.next();
    if infile_wrap == None {
        exit_help();
    }
    let infile_temp = String::from(infile_wrap.unwrap());
    let infile = Path::new(&infile_temp);
    let outfile_wrap = args.next();
    if outfile_wrap == None {
        exit_help();
    }
    let outfile_temp = String::from(outfile_wrap.unwrap());
    let outfile = Path::new(&outfile_temp);
    let mut fontsizes: Vec<i32> = Vec::new();
    let mut tsize = 512usize;
    let mut padding = 3i32;
    while let Some(arg) = args.next() {
        match &arg[..] {
            "-h" | "--help" => { exit_help(); },
            "-v" | "--version" => { exit_version(); },
            "-s" | "--size" => {
                if let Some(value_string) = args.next() {
                    if let Ok(value) = value_string.parse::<i32>() {
                        fontsizes.push(value);
                    }
                    else {
                        exit_help();
                    }
                }
                else {
                    exit_help();
                }
            },
            "-d" | "--dim" => {
                if let Some(value_string) = args.next() {
                    if let Ok(value) = value_string.parse::<usize>() {
                        tsize = value;
                    }
                    else {
                        exit_help();
                    }
                }
                else {
                    exit_help();
                }
            },
            "-p" | "--padding" => {
                if let Some(value_string) = args.next() {
                    if let Ok(value) = value_string.parse::<i32>() {
                        padding = value;
                    }
                    else {
                        exit_help();
                    }
                }
                else {
                    exit_help();
                }
            },
            &_ => {
                exit_help();
            },
        }
    }
    if fontsizes.len() == 0 {
        fontsizes.push(20);
    }

    // initialize FreeType
    let ft = freetype::Library::init().unwrap();
    let face = ft.new_face(infile.to_str().unwrap(),0).unwrap();
    let mut image_character_sets: Vec<ImageCharacterSet> = Vec::new();
    let mut image_characters: Vec<ImageCharacter> = Vec::new();
    for fontsize in fontsizes.iter() {
        face.set_char_size((fontsize * 64) as isize,0,72,0).unwrap();
        let mut min = 0i32;
        let mut max = 0i32;
        for set in CHARACTERS.iter() {
            for n in set.0..set.1 {
                face.load_char(n as usize,freetype::face::LoadFlag::RENDER).unwrap();
                let glyph = face.glyph();
                let bitmap = glyph.bitmap();
                let width = bitmap.width();
                let height = bitmap.rows();
                let buffer = bitmap.buffer();
                let metrics = glyph.metrics();
                let bx = metrics.horiBearingX >> 6;
                let by = metrics.horiBearingY >> 6;
                let a = metrics.horiAdvance >> 6;
                println!("{:04X}: {}x{}, bearing {},{}, advance {}",n,width,height,bx,by,a);
                let mut cutout = Mat::<u8>::new(Vec2::<usize> {
                    x: (width + 2 * padding) as usize,
                    y: (height + 2 * padding) as usize
                });
                for y in 0..height {
                    for x in 0..width {
                        let b = buffer[(y * width + x) as usize];
                        cutout[((x + padding) as usize,(y + padding) as usize)] = b;
                    }
                }
                image_characters.push(ImageCharacter {
                    size: *fontsize as u32,
                    n: n,
                    image: cutout,
                    bearing: Vec2::<i32>::new(bx as i32,by as i32),
                    advance: a as i32,
                });
                if -(by as i32) < min {
                    min = -(by as i32);
                }
                if -(by as i32) + (height as i32) > max {
                    max = -(by as i32) + (height as i32);
                }
            }
        }
        image_character_sets.push(ImageCharacterSet {
            size: *fontsize as u32,
            min: min,
            max: max,
        });
    }

    //Command::new("sh").arg("-c").arg(format!("rm output.png")).output().expect("unable to remove output.png");

    // sort image characters by height
    image_characters.sort_by(|a,b| b.image.size.y.cmp(&a.image.size.y));

    // prepare atlas and character set structs
    let mut image = Mat::<Mapped>::new(Vec2::<usize> { x: tsize,y: tsize, });
    let mut character_sets: Vec<CharacterSet> = Vec::new();
    for ics in image_character_sets.iter() {
        character_sets.push(CharacterSet {
            size: ics.size,
            min: ics.min,
            max: ics.max,
            characters: Vec::new(),
        });
    }

    // enter all characters
    let total_i = image_characters.len();
    let mut i = 0u32;
    for ch in image_characters.iter() {
        println!("Character {} / {} (size {}, code {})",i,total_i,ch.size,ch.n);
        let mut p = Vec2::<isize> { x: 0,y: 0, };
        if !find_rect(&ch.image,&image,&mut p) {
            if find_empty(&ch.image.size,&image,&mut p) {
                for y in 0..ch.image.size.y {
                    for x in 0..ch.image.size.x {
                        image[(p.x as usize + x,p.y as usize + y)] = Mapped { used: true,value: ch.image[(x,y)], };
                    }
                }
            }
            else {
                println!("Unable to fit all characters onto {}x{}.",tsize,tsize);
                return;
            }
        }
        let r = rect!(
            p.x as i32 + padding,
            p.y as i32 + padding,
            ch.image.size.x as i32 - 2 * padding,
            ch.image.size.y as i32 - 2 * padding
        );
        for cs in character_sets.iter_mut() {
            if cs.size == ch.size {
                cs.characters.push(Character {
                    n: ch.n,
                    r: r,
                    bearing: ch.bearing,
                    advance: ch.advance,
                });
            }
        }
        i += 1;
    }

    // save to file
    let mut file = File::create(outfile.file_name().unwrap().to_str().unwrap()).expect("cannot create file");
    let mut buffer: Vec<u8> = Vec::new();

    buffer.push(0x45);
    buffer.push(0x46);
    buffer.push(0x4E);
    buffer.push(0x54);
    buffer.push(0x30);
    buffer.push(0x30);
    buffer.push(0x33);
    buffer.push(0x00);
    push_u32(&mut buffer,image.size.x as u32);  // texture atlas width
    push_u32(&mut buffer,image.size.y as u32);  // texture atlas height
    push_u32(&mut buffer,character_sets.len() as u32);  // number of font sizes in texture
    for character_set in character_sets.iter() {
        push_u32(&mut buffer,character_set.size);  // font size
        push_i32(&mut buffer,character_set.max - character_set.min);  // font height (actual pixels needed)
        push_i32(&mut buffer,-character_set.min);  // font Y bearing
        push_u32(&mut buffer,character_set.characters.len() as u32);  // number of characters
        for ch in &character_set.characters {
            push_u32(&mut buffer,ch.n);
            push_i32(&mut buffer,ch.r.o.x);
            push_i32(&mut buffer,ch.r.o.y);
            push_i32(&mut buffer,ch.r.s.x);
            push_i32(&mut buffer,ch.r.s.y);
            push_i32(&mut buffer,ch.bearing.x);
            push_i32(&mut buffer,ch.bearing.y);
            push_i32(&mut buffer,ch.advance);
            //println!("{:04X}: {},{} ({}x{}); {},{}; {}",ch.n,ch.r.o.x,ch.r.o.y,ch.r.s.x,ch.r.s.y,ch.offset.x,ch.offset.y,ch.advance);
        }    
    }
    for y in 0..image.size.y {
        for x in 0..image.size.x {
            buffer.push(image[(x,y)].value);
        }
    }
    file.write_all(&buffer).expect("cannot write");

    // also write debug output image
    let mut file = File::create("debug.bmp").expect("what?");
    let mut debug_image = Mat::<pixelformat::BGRA8UN>::new(image.size);
    for y in 0..image.size.y {
        for x in 0..image.size.x {
            let b = image[(x,y)].value;
            debug_image[(x,y)].set(b,b,b,255);
        }
    }
    let debug_buffer = bmp::encode(&debug_image).expect("what?");
    file.write_all(&debug_buffer).expect("what?");
}
