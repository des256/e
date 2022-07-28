use crate::*;

// Inflate algorithm
const LITLEN_LENGTH: [u16; 29] = [3,4,5,6,7,8,9,10,11,13,15,17,19,23,27,31,35,43,51,59,67,83,99,115,131,163,195,227,258];
const LITLEN_EXTRA: [u8; 29] = [0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0];
const DIST_DIST: [u16; 30] = [1,2,3,4,5,7,9,13,17,25,33,49,65,97,129,193,257,385,513,769,1025,1537,2049,3073,4097,6145,8193,12289,16385,24577];
const DIST_EXTRA: [u8; 30] = [0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13];
const HCORD: [usize; 19] = [16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15];

#[derive(Copy,Clone)]
enum Type {
    L1,
    C1,
    L2,
    C2,
    L4,
    C4,
    L8,
    RGB8,
    C8,
    LA8,
    RGBA8,
    L16,
    RGB16,
    LA16,
    RGBA16,
}

// grayscale distributions
const GRAY2: [f32; 4] = [0.0,0.33333333,0.66666667,1.0];

const GRAY4: [f32; 16] = [
    0.0,0.06666667,0.13333333,0.2,
    0.26666667,0.33333333,0.4,0.46666667,
    0.53333333,0.6,0.66666667,0.73333333,
    0.8,0.86666667,0.93333333,1.0
];

const TABLE: usize = 8;  // 8 seems to be a good balance
const TABLE_SIZE: usize = 1 << TABLE;

fn bit_reverse(value: u32,width: u32) -> u32 {
    let mut result: u32 = 0;
    for i in 0..width {
        let bit: u32 = (value >> i) & 1;
        result |= bit << (width - i - 1);
    }
    result
}

fn insert_code(tables: &mut Vec<[i16; TABLE_SIZE]>,ofs: u32,code: u16,length: u8) -> u32 {
    let shift = 32 - TABLE;
    if (length as usize) > TABLE {
        let pos: usize = ((ofs >> shift) & ((TABLE_SIZE - 1) as u32)) as usize;
        let p = bit_reverse(pos as u32,TABLE as u32) as usize;
        let mut n: i16 = tables.len() as i16;
        if tables[0][p] == 0 {
            tables.push([0i16; TABLE_SIZE]);
            tables[0][p] = -n;
        }
        else {
            n = -tables[0][p];
        }
        let shift = 32 - TABLE - TABLE;
        let pos = ((ofs >> shift) & ((TABLE_SIZE - 1) as u32)) as usize;
        let count = TABLE_SIZE >> (length - TABLE as u8) as usize;
        for i in pos..pos + count {
            let p = bit_reverse(i as u32,TABLE as u32) as usize;
            tables[n as usize][p] = ((code << 5) | (length as u16)) as i16;
        }
        (count << shift) as u32				
    }
    else {
        let pos = ((ofs >> shift) & ((TABLE_SIZE - 1) as u32)) as usize;
        let count = TABLE_SIZE >> length as usize;
        for i in pos..pos + count {
            let p = bit_reverse(i as u32,TABLE as u32) as usize;
            tables[0][p] = ((code << 5) | (length as u16)) as i16;
        }
        (count << shift) as u32
    }
}

fn create_huffman_tables(lengths: &[u8]) -> Vec<[i16; TABLE_SIZE]> {
    let mut tables: Vec<[i16; TABLE_SIZE]> = Vec::new();
    tables.push([0i16; TABLE_SIZE]);
    let mut ofs: u64 = 0;
    for i in 1..25 {
        for k in 0..lengths.len() {
            if lengths[k] == i {
                let size = insert_code(&mut tables,ofs as u32,k as u16,lengths[k]);
                ofs += size as u64;
            }
        }
    }
    tables
}

struct ZipReader<'a> {
    block: &'a [u8],
    rp: usize,
    bit: u32,
    cache: u32,
}

impl<'a> ZipReader<'a> {
    // This currently entirely ignores the ZLIB wrapper that goes around the INFLATE blocks; this might not always be a good thing
    fn new(block: &'a [u8]) -> ZipReader<'a> {
        ZipReader {
            block: block,
            rp: 6,
            bit: 32,
            cache: ((block[2] as u32) | ((block[3] as u32) << 8) | ((block[4] as u32) << 16) | ((block[5] as u32) << 24)) as u32,
        }        
    }

    fn align(&mut self) -> usize {
        if self.bit < 32 {
            self.bit = 32;
            self.rp - 3
        }
        else {
            self.rp - 4
        }
    }

    fn set(&mut self,rp: usize) {
        self.rp = rp;
        self.cache = (self.block[self.rp] as u32) |
            ((self.block[self.rp + 1] as u32) << 8) |
            ((self.block[self.rp + 2] as u32) << 16) |
            ((self.block[self.rp + 3] as u32) << 24);
        self.rp += 4;
        self.bit = 32;
    }

    fn read_bits(&mut self,n: u32) -> Option<u32> {
        let result: u32 = self.cache & ((1 << n) - 1);
        self.cache >>= n;
        self.bit -= n;
        while self.bit <= 24 {
            if self.rp >= self.block.len() {
                return None;
            }
            self.cache |= (self.block[self.rp] as u32) << self.bit;
            self.rp += 1;
            self.bit += 8;
        }
        Some(result)
    }

    fn read_symbol(&mut self,prefix: &Vec<[i16; TABLE_SIZE]>) -> Option<u32> {

        let mut n: usize = 0;
        let mut index = (self.cache & (TABLE_SIZE - 1) as u32) as usize;
        let mut stuff = prefix[n][index];
        let mut already_shifted = 0;
        while stuff < 0 {
            self.cache >>= TABLE;
            self.bit -= TABLE as u32;
            while self.bit <= 24 {
                if self.rp >= self.block.len() {
                    return None;
                }
                self.cache |= (self.block[self.rp] as u32) << self.bit;
                self.rp += 1;
                self.bit += 8;
            }
            already_shifted += TABLE;
            n = (-stuff) as usize;
            index = (self.cache & (TABLE_SIZE - 1) as u32) as usize;
            stuff = prefix[n][index];
        }
        let symbol = stuff >> 5;
        let length = (stuff & 31) - already_shifted as i16;
        self.cache >>= length;
        self.bit -= length as u32;
        while self.bit <= 24 {
            if self.rp >= self.block.len() {
                return None;
            }
            self.cache |= (self.block[self.rp] as u32) << self.bit;
            self.rp += 1;
            self.bit += 8;
        }
        Some(symbol as u32)
    }
}

fn inflate(src: &[u8],inflated_size: u32) -> Option<Vec<u8>> {

    //println!("PNG: ZIP INFLATE: zlib header: {:02X} {:02X}; inflated size supposed to be {}, source size {}",src[0],src[1],inflated_size,src.len());

    let mut dst: Vec<u8> = vec![0; inflated_size as usize];
    let mut reader = ZipReader::new(&src);
    let mut dp: usize = 0;

    // create default litlen table
    let mut lengths: [u8; 288] = [0; 288];
    for i in 0..144 {
        lengths[i] = 8;
    }
    for i in 144..256 {
        lengths[i] = 9;
    }
    for i in 256..280 {
        lengths[i] = 7;
    }
    for i in 280..288 {
        lengths[i] = 8;
    }

    let default_hlitlen_tables = create_huffman_tables(&lengths);
    #[allow(unused_assignments)]
    let mut current_hlitlen_tables: Vec<[i16; TABLE_SIZE]> = Vec::new();

    // create default dist table
    let lengths: [u8; 32] = [5; 32];
    let default_hdist_tables = create_huffman_tables(&lengths);
    #[allow(unused_assignments)]
    let mut current_hdist_tables: Vec<[i16; TABLE_SIZE]> = Vec::new();

    let mut hlitlen_tables: &Vec<[i16; TABLE_SIZE]> = &default_hlitlen_tables;
    let mut hdist_tables: &Vec<[i16; TABLE_SIZE]> = &default_hdist_tables;

    // main loop
    let mut is_final = false;
    while !is_final {

        // get final block and type bits
        match reader.read_bits(1) {
            Some(value) => { if value == 1 { is_final = true; } else { is_final = false; } },
            None => {
                //println!("PNG: ZIP INFLATE: unable to read final block bit");
                return None;
            },
        }
        let block_type = match reader.read_bits(2) {
            Some(value) => { value },
            None => {
                //println!("PNG: ZIP INFLATE: unable to read type bits");
                return None;
            },
        };

        //println!("PNG: ZIP INFLATE: block final {}, type {}",is_final,block_type);

        // process uncompressed data
        match block_type {
            0 => {
                let mut sp = reader.align();
                let length = (((src[sp + 1] as u16) << 8) | (src[sp] as u16)) as usize;
                sp += 4;
                if (sp + length > src.len()) || (dp + length > dst.len()) {
                    //println!("PNG: ZIP INFLATE: uncompressed data out of range");
                    return None;
                }
                dst[dp..dp + length].copy_from_slice(&src[sp..sp + length]);
                sp += length;
                dp += length;
                reader.set(sp);
            },
            1 => {
                hlitlen_tables = &default_hlitlen_tables;
                hdist_tables = &default_hdist_tables;
            },
            2 => {
                // get table metrics
                let hlit = match reader.read_bits(5) {
                    Some(value) => { value + 257 },
                    None => {
                        //println!("PNG: ZIP INFLATE: unable to read hlit bits");
                        return None;
                    },
                } as usize;
                let hdist = match reader.read_bits(5) {
                    Some(value) => { value + 1 },
                    None => {
                        //println!("PNG: ZIP INFLATE: unable to read hdist bits");
                        return None;
                    },
                } as usize;
                let hclen = match reader.read_bits(4) {
                    Some(value) => { value + 4 },
                    None => {
                        //println!("PNG: ZIP INFLATE: unable to read hclen bits");
                        return None;
                    },
                };

                // get length codes
                let mut lengths: [u8; 20] = [0; 20];
                for i in 0..hclen {
                    lengths[HCORD[i as usize]] = match reader.read_bits(3) {
                        Some(value) => { value },
                        None => {
                            //println!("PNG: ZIP INFLATE: unable to read length code bits");
                            return None;
                        },
                    } as u8;
                }
                let hctree_tables = create_huffman_tables(&lengths);

                // no really, get length codes
                let mut lengths: [u8; 320] = [0; 320];
                let mut ll: usize = 0;
                while ll < hlit + hdist {
                    let code = match reader.read_symbol(&hctree_tables) {
                        Some(value) => { value },
                        None => {
                            //println!("PNG: ZIP INFLATE: unable to read code symbol");
                            return None;
                        },
                    };
                    if code == 16 {
                        let length = match reader.read_bits(2) {
                            Some(value) => { value + 3 },
                            None => {
                                //println!("PNG: ZIP INFLATE: unable to read 2 length bits");
                                return None;
                            },
                        };
                        for _i in 0..length { // TODO: for loop might be expressed differently in rust
                            lengths[ll] = lengths[ll - 1];
                            ll += 1;
                        }
                    }
                    else if code == 17 {
                        let length = match reader.read_bits(3) {
                            Some(value) => { value + 3 },
                            None => {
                                //println!("PNG: ZIP INFLATE: unable to read 3 length bits");
                                return None;
                            },
                        };
                        for _i in 0..length { // TODO: for loop might be expressed differently in rust
                            lengths[ll] = 0;
                            ll += 1;
                        }
                    }
                    else if code == 18 {
                        let length = match reader.read_bits(7) {
                            Some(value) => { value + 11 },
                            None => {
                                //println!("PNG: ZIP INFLATE: unable to read 7 length bits");
                                return None;
                            },
                        };
                        for _i in 0..length { // TODO: for loop might be expressed differently in rust
                            lengths[ll] = 0;
                            ll += 1;
                        }
                    }
                    else {
                        lengths[ll] = code as u8;
                        ll += 1;
                    }
                }

                current_hlitlen_tables = create_huffman_tables(&lengths[0..hlit]);
                current_hdist_tables = create_huffman_tables(&lengths[hlit..hlit + hdist]);

                hlitlen_tables = &current_hlitlen_tables;
                hdist_tables = &current_hdist_tables;
            },
            3 => {
                if is_final {
                    break;
                }
                else {
                    //println!("PNG: ZIP INFLATE: block type 3 not supported");
                    return None;
                }
            },
            _ => { },
        }
        if (block_type == 1) || (block_type == 2) {
            // read them codes
            while dp < dst.len() {
                let mut code = match reader.read_symbol(&hlitlen_tables) {
                    Some(value) => { value },
                    None => {
                        //println!("PNG: ZIP INFLATE: unable to read code symbol");
                        return None;
                    },
                };
                if code < 256 {
                    dst[dp] = code as u8;
                    dp += 1;
                }
                else if code == 256 {
                    break;
                }
                else {
                    // get lit/len length and extra bit entries
                    code -= 257;
                    let mut length = LITLEN_LENGTH[code as usize] as usize;
                    let extra = LITLEN_EXTRA[code as usize] as u32;

                    // read extra bits
                    if extra > 0 {
                        length += match reader.read_bits(extra) {
                            Some(value) => { value },
                            None => {
                                //println!("PNG: ZIP INFLATE: unable to read extra length bits");
                                return None;
                            },
                        } as usize;
                    }

                    // get dist length and extra bit entries
                    code = match reader.read_symbol(&hdist_tables) {
                        Some(value) => { value },
                        None => {
                            //println!("PNG: ZIP INFLATE: unable to read extra dist length");
                            return None;
                        },
                    };
                    let mut dist = DIST_DIST[code as usize] as usize;
                    let extra = DIST_EXTRA[code as usize] as u32;

                    // read extra bits
                    if extra > 0 {
                        dist += match reader.read_bits(extra) {
                            Some(value) => { value },
                            None => {
                                //println!("PNG: ZIP INFLATE: unable to read extra bits");
                                return None;
                            },
                        } as usize;
                    }

                    // copy block
                    if dp + length > dst.len() {
                        length = dst.len() - dp;
                        //return None;
                    }
                    if dist > dp {
                        //println!("PNG: ZIP INFLATE: distance too large");
                        return None;
                    }
                    if dp + length - dist > dst.len() {
                        //println!("PNG: ZIP INFLATE: source block too big");
                        return None;
                    }
                    for i in 0..length {
                        dst[dp + i] = dst[dp - dist + i];
                    }
                    dp += length;
                }
            }
        }
    }
    Some(dst)
}

fn unfilter(src: &[u8],height: usize,stride: usize,bpp: usize) -> Vec<u8> {
    let mut dst: Vec<u8> = vec![0; stride * height * bpp];
    let mut sp: usize = 0;
    let mut dp: usize = 0;
    for y in 0..height {
        let ftype = src[sp];
        sp += 1;
        for x in 0..stride {
            let mut s = src[sp] as i32;
            sp += 1;
            let a: i32 = if x >= bpp { dst[dp - bpp] as i32 } else { 0 };
            let b: i32 = if y >= 1 { dst[dp - stride] as i32 } else { 0 };
            let c: i32 = if (y >= 1) && (x >= bpp) { dst[dp - stride - bpp] as i32 } else { 0 };
            s += match ftype {
                0 => { 0 },
                1 => { a },
                2 => { b },
                3 => { (a + b) >> 1 },
                4 => {
                    let d: i32 = a + b - c;
                    let da: i32 = d - a;
                    let pa: i32 = if da < 0 { -da } else { da };
                    let db: i32 = d - b;
                    let pb: i32 = if db < 0 { -db } else { db };
                    let dc: i32 = d - c;
                    let pc: i32 = if dc < 0 { -dc } else { dc };
                    if (pa <= pb) && (pa <= pc) { a } else if pb <= pc { b } else { c }
                },
                _ => { 0 },
            };
            if s >= 256 { s -= 256 };
            if s < 0 { s += 256 };
            dst[dp] = s as u8;
            dp += 1;
        }
    }
    dst
}

fn clampf(v: f32,min: f32,max: f32) -> f32 {
    if v < min {
        min
    }
    else if v > max {
        max
    }
    else {
        v
    }
}

fn set_lf<T: Pixel>(p: &mut T,l: f32,gamma: f32) {
    let ul = (clampf(l.powf(gamma),0.0,1.0) * 255.0) as u8;
    p.set(ul,ul,ul,255);
}

fn set_rgbaf<T: Pixel>(p: &mut T,r: f32,g: f32,b: f32,a: f32,gamma: f32) {
    let ur = (clampf(r.powf(gamma),0.0,1.0) * 255.0) as u8;
    let ug = (clampf(g.powf(gamma),0.0,1.0) * 255.0) as u8;
    let ub = (clampf(b.powf(gamma),0.0,1.0) * 255.0) as u8;
    let ua = (clampf(a.powf(gamma),0.0,1.0) * 255.0) as u8;
    p.set(ur,ug,ub,ua);
}

fn set_c<T: Pixel>(p: &mut T,c: T,gamma: f32) {
    let (r,g,b,a) = c.get();
    let r = (r as f32) / 255.0;
    let g = (g as f32) / 255.0;
    let b = (b as f32) / 255.0;
    let a = (a as f32) / 255.0;
    let ur = (clampf(r.powf(gamma),0.0,1.0) * 255.0) as u8;
    let ug = (clampf(g.powf(gamma),0.0,1.0) * 255.0) as u8;
    let ub = (clampf(b.powf(gamma),0.0,1.0) * 255.0) as u8;
    let ua = (clampf(a.powf(gamma),0.0,1.0) * 255.0) as u8;
    p.set(ur,ug,ub,ua);
}

fn decode_pixels<T: Pixel>(dst: &mut Mat<T>,src: &[u8],width: usize,height: usize,stride: usize,x0: usize,y0: usize,dx: usize,dy: usize,itype: Type,palette: &[T; 256],gamma: f32) {
    let mut sp = 0;
    match itype {
        Type::L1 => {
            for y in 0..height {
                for x in 0..(width / 8) {
                    let d = src[sp];
                    sp += 1;
                    for i in 0..8 {
                        let l = if(d & (0x80 >> i)) != 0 { 1.0 } else { 0.0 };
                        set_lf(&mut dst[(y0 + y * dy) * stride + x0 + (x * 8 + i) * dx],l,gamma);
                    }
                }
                if (width & 7) != 0 {
                    let d = src[sp];
                    sp += 1;
                    for i in 0..8 {
                        let l = if(d & (0x80 >> i)) != 0 { 1.0 } else { 0.0 };
                        set_lf(&mut dst[(y0 + y * dy) * stride + x0 + ((width & 0xFFFFFFF8) + i) * dx],l,gamma);
                    }
                }
            }
        },
        Type::C1 => {
            for y in 0..height {
                for x in 0..(width / 8) {
                    let d = src[sp];
                    sp += 1;
                    for i in 0..8 {
                        let c = if (d & (0x80 >> i)) != 0 { palette[1] } else { palette[0] };
                        set_c(&mut dst[(y0 + y * dy) * stride + x0 + (x * 8 + i) * dx],c,gamma);
                    }
                }
                if (width & 7) != 0 {
                    let d = src[sp];
                    sp += 1;
                    for i in 0..(width & 7) {
                        let c = if (d & (0x80 >> i)) != 0 { palette[1] } else { palette[0] };
                        set_c(&mut dst[(y0 + y * dy) * stride + x0 + ((width & 0xFFFFFFF8) + i) * dx],c,gamma);
                    }
                }
            }
        },
        Type::L2 => {
            for y in 0..height {
                for x in 0..(width / 4) {
                    let d = src[sp];
                    sp += 1;
                    for i in 0..4 {
                        set_lf(&mut dst[(y0 + y * dy) * stride + x0 + (x * 4 + i) * dx],GRAY2[((d >> ((3 - i) * 2)) & 3) as usize],gamma);
                    }
                }
                if(width & 3) != 0 {
                    let d = src[sp];
                    sp += 1;
                    for i in 0..(width & 3) {
                        set_lf(&mut dst[(y0 + y * dy) * stride + x0 + ((width & 0xFFFFFFFC) + i) * dx],GRAY2[((d >> ((3 - i) * 2)) & 3) as usize],gamma);
                    }
                }
            }
        },
        Type::C2 => {
            for y in 0..height {
                for x in 0..(width / 4) {
                    let d = src[sp];
                    sp += 1;
                    for i in 0..4 {
                        set_c(&mut dst[(y0 + y * dy) * stride + x0 + (x * 4 + i) * dx],palette[((d >> ((3 - i) * 2)) & 3) as usize],gamma);
                    }
                }
                if(width & 3) != 0 {
                    let d = src[sp];
                    sp += 1;
                    for i in 0..(width & 3) {
                        set_c(&mut dst[(y0 + y * dy) * stride + x0 + ((width & 0xFFFFFFFC) + i) * dx],palette[((d >> ((3 - i) * 2)) & 3) as usize],gamma);
                    }
                }
            }
        },
        Type::L4 => {
            for y in 0..height {
                for x in 0..(width / 2) {
                    let d = src[sp];
                    sp += 1;
                    for i in 0..2 {
                        set_lf(&mut dst[(y0 + y * dy) * stride + x0 + (x * 2 + i) * dx],GRAY4[((d >> ((1 >> i) * 4)) & 15) as usize],gamma);
                    }
                }
                if (width & 1) != 0 {
                    set_lf(&mut dst[(y0 + y * dy) * stride + x0 + (width & 0xFFFFFFFE) * dx],GRAY4[(src[sp] >> 4) as usize],gamma);
                    sp += 1;
                }
            }
        },
        Type::C4 => {
            for y in 0..height {
                for x in 0..(width / 2) {
                    let d = src[sp];
                    sp += 1;
                    for i in 0..2 {
                        set_c(&mut dst[(y0 + y * dy) * stride + x0 + (x * 2 + i) * dx],palette[((d >> ((1 >> i) * 4)) & 15) as usize],gamma);
                    }
                }
                if (width & 1) != 0 {
                    set_c(&mut dst[(y0 + y * dy) * stride + x0 + (width & 0xFFFFFFFE) * dx],palette[(src[sp] >> 4) as usize],gamma);
                    sp += 1;
                }
            }
        },
        Type::L8 => {
            for y in 0..height {
                for x in 0..width {
                    let l = (src[sp] as f32) / 255.0;
                    sp += 1;
                    set_lf(&mut dst[(y0 + y * dy) * stride + x0 + x * dx],l,gamma);
                }
            }
        },
        Type::RGB8 => {
            for y in 0..height {
                for x in 0..width {
                    let r = (src[sp] as f32) / 255.0;
                    let g = (src[sp + 1] as f32) / 255.0;
                    let b = (src[sp + 2] as f32) / 255.0;
                    sp += 3;
                    set_rgbaf(&mut dst[(y0 + y * dy) * stride + x0 + x * dx],r,g,b,1.0,gamma);
                }
            }
        },
        Type::C8 => {
            for y in 0..height {
                for x in 0..width {
                    let c = src[sp];
                    sp += 1;
                    set_c(&mut dst[(y0 + y * dy) * stride + x0 + x * dx],palette[c as usize],gamma);
                }
            }
        },
        Type::LA8 => {
            for y in 0..height {
                for x in 0..width {
                    let l = (src[sp] as f32) / 255.0;
                    let a = (src[sp + 1] as f32) / 255.0;
                    sp += 2;
                    set_rgbaf(&mut dst[(y0 + y * dy) * stride + x0 + x * dx],l,l,l,a,gamma);
                }
            }
        },
        Type::RGBA8 => {
            for y in 0..height {
                for x in 0..width {
                    let r = (src[sp] as f32) / 255.0;
                    let g = (src[sp + 1] as f32) / 255.0;
                    let b = (src[sp + 2] as f32) / 255.0;
                    let a = (src[sp + 3] as f32) / 255.0;
                    sp += 4;
                    set_rgbaf(&mut dst[(y0 + y * dy) * stride + x0 + x * dx],r,g,b,a,gamma);
                }
            }
        },
        Type::L16 => {
            for y in 0..height {
                for x in 0..width {
                    let l = (src[sp] as f32) / 255.0;
                    sp += 2;
                    set_lf(&mut dst[(y0 + y * dy) * stride + x0 + x * dx],l,gamma);
                }
            }
        },
        Type::RGB16 => {
            for y in 0..height {
                for x in 0..width {
                    let r = (src[sp] as f32) / 255.0;
                    let g = (src[sp + 2] as f32) / 255.0;
                    let b = (src[sp + 4] as f32) / 255.0;
                    sp += 6;
                    set_rgbaf(&mut dst[(y0 + y * dy) * stride + x0 + x * dx],r,g,b,1.0,gamma);
                }
            }
        },
        Type::LA16 => {
            for y in 0..height {
                for x in 0..width {
                    let l = (src[sp] as f32) / 255.0;
                    let a = (src[sp + 2] as f32) / 255.0;
                    sp += 4;
                    set_rgbaf(&mut dst[(y0 + y * dy) * stride + x0 + x * dx],l,l,l,a,gamma);
                }
            }
        },
        Type::RGBA16 => {
            for y in 0..height {
                for x in 0..width {
                    let r = (src[sp] as f32) / 255.0;
                    let g = (src[sp + 2] as f32) / 255.0;
                    let b = (src[sp + 4] as f32) / 255.0;
                    let a = (src[sp + 6] as f32) / 255.0;
                    sp += 8;
                    set_rgbaf(&mut dst[(y0 + y * dy) * stride + x0 + x * dx],r,g,b,a,gamma);
                }
            }
        },
    }
}

fn from_be16(src: &[u8]) -> u16 {
    ((src[0] as u16) << 8) | (src[1] as u16)
}

fn from_be32(src: &[u8]) -> u32 {
    ((src[0] as u32) << 24) | ((src[1] as u32) << 16) | ((src[2] as u32) << 8) | (src[3] as u32)
}

pub fn test(src: &[u8]) -> Option<(u32,u32)> {
    if (src[0] == 0x89) && (src[1] == 0x50) && (src[2] == 0x4E) && (src[3] == 0x47) && (src[4] == 0x0D) && (src[5] == 0x0A) && (src[6] == 0x1A) && (src[7] == 0x0A) {
        let mut sp: usize = 8;
        while sp < src.len() {
            let chunk_length = from_be32(&src[sp..sp + 4]) as usize;
            sp += 4;
            let chunk_type = from_be32(&src[sp..sp + 4]);
            sp += 4;
            if chunk_type == 0x49484452 { // IHDR
                let width = from_be32(&src[sp..sp + 4]);
                sp += 4;
                let height = from_be32(&src[sp..sp + 4]);
                sp += 4;
                let t = from_be16(&src[sp..sp + 2]);
                match t {
                    0x0100 | 0x0103 | 0x0200 | 0x0203 | 0x0400 | 0x0403 | 0x0800 | 0x0802 | 0x0803 | 0x0804 | 0x0806 | 0x1000 | 0x1002 | 0x1004 | 0x1006 => { return Some((width,height)); },
                    _ => { return None; },
                }
            }
            else if chunk_type == 0x49454E44 { // IEND
                break;
            }
            else {
                sp += chunk_length;
            }
            sp += 4;
        }
    }
    None
}

pub fn decode<T: Pixel + Default>(src: &[u8]) -> Option<Mat<T>> {
    if (src[0] != 0x89) ||
        (src[1] != 0x50) ||
        (src[2] != 0x4E) ||
        (src[3] != 0x47) ||
        (src[4] != 0x0D) ||
        (src[5] != 0x0A) ||
        (src[6] != 0x1A) ||
        (src[7] != 0x0A) {
        return None;
    }
    let mut sp: usize = 8;
    let mut width: u32 = 0;
    let mut height: u32 = 0;
    let mut itype = Type::L1;
#[allow(unused_assignments)]
    let mut compression: u8 = 0;
#[allow(unused_assignments)]
    let mut filter: u8 = 0;
    let mut interlace: u8 = 0;
    let mut stride: u32 = 0;
    let mut bpp: usize = 0;
    let mut need_plte = false;
    let mut plte_present = false;
    let mut zipped_data: Vec<u8> = vec![0; src.len()];
    let mut dp: usize = 0;
    let mut idat_found = false;
    let mut iend_found = false;
    let mut palette = [T::default(); 256];
    let mut _background = T::default();
    let mut gamma: f32 = 1.0;
    while sp < src.len() {
        let chunk_length = from_be32(&src[sp..sp + 4]) as usize;
        sp += 4;
        let chunk_type = from_be32(&src[sp..sp + 4]);
        sp += 4;
        match chunk_type {
            0x49484452 => { // IHDR
                width = from_be32(&src[sp..]);
                height = from_be32(&src[sp + 4..]);
                let itype_code = from_be16(&src[sp + 8..]);
                compression = src[sp + 10];
                filter = src[sp + 11];
                interlace = src[sp + 12];
                if (width >= 65536) ||
                    (height >= 65536) ||
                    (compression != 0) ||
                    (filter != 0) ||
                    (interlace > 1) {
                    //println!("PNG: header sanity check failed");
                    return None;
                }
                itype = match itype_code {
                    0x0100 => Type::L1,
                    0x0103 => Type::C1,
                    0x0200 => Type::L2,
                    0x0203 => Type::C2,
                    0x0400 => Type::L4,
                    0x0403 => Type::C4,
                    0x0800 => Type::L8,
                    0x0802 => Type::RGB8,
                    0x0803 => Type::C8,
                    0x0804 => Type::LA8,
                    0x0806 => Type::RGBA8,
                    0x1000 => Type::L16,
                    0x1002 => Type::RGB16,
                    0x1004 => Type::LA16,
                    0x1006 => Type::RGBA16,
                    _ => {
                        //println!("PNG: unknown itype {:04X}",itype_code);
                        return None;
                    },
                };
                match itype {
                    Type::L1 => { stride = (width + 7) / 8; bpp = 1; },
                    Type::C1 => { stride = (width + 7) / 8; bpp = 1; need_plte = true; },
                    Type::L2 => { stride = (width + 3) / 4; bpp = 1; },
                    Type::C2 => { stride = (width + 3) / 4; bpp = 1; need_plte = true; },
                    Type::L4 => { stride = (width + 1) / 2; bpp = 1; },
                    Type::C4 => { stride = (width + 1) / 2; bpp = 1; need_plte = true; },
                    Type::L8 => { stride = width; bpp = 1; },
                    Type::RGB8 => { stride = width * 3; bpp = 3; },
                    Type::C8 => { stride = width; bpp = 1; need_plte = true; },
                    Type::LA8 => { stride = width * 2; bpp = 2; },
                    Type::RGBA8 => { stride = width * 4; bpp = 4; },
                    Type::L16 => { stride = width * 2; bpp = 2; },
                    Type::RGB16 => { stride = width * 6; bpp = 6; },
                    Type::LA16 => { stride = width * 2; bpp = 4; },
                    Type::RGBA16 => { stride = width * 4; bpp = 8; },
                }
                sp += chunk_length;
            },
            0x49444154 => { // IDAT
                zipped_data[dp..dp + chunk_length].copy_from_slice(&src[sp..sp + chunk_length]);
                sp += chunk_length;
                dp += chunk_length;
                idat_found = true;
            },
            0x49454E44 => { // IEND
                iend_found = true;
                break;
            },
            0x504C5445 => { // PLTE
                plte_present = true;
                if chunk_length > 768 {
                    //println!("PNG: PLTE chunk too big ({})",chunk_length);
                    return None;
                }
                for i in 0..(chunk_length / 3) {
                    let r = src[sp];
                    let g = src[sp + 1];
                    let b = src[sp + 2];
                    sp += 3;
                    palette[i].set(r,g,b,255);
                }
            },
            0x624B4744 => { // bKGD
                match itype {
                    Type::C1 | Type::C2 | Type::C4 | Type::C8 => {
                        _background = palette[src[sp] as usize];
                    },
                    Type::L1 | Type::L2 | Type::L4 | Type::L8 | Type::LA8 | Type::L16 | Type::LA16 => {
                        let level = src[sp];
                        _background.set(level,level,level,255);
                    },
                    _ => {
                        let r = src[sp];
                        let g = src[sp + 2];
                        let b = src[sp + 4];
                        _background.set(r,g,b,255);
                    },
                }
                sp += chunk_length;
            },
            0x74524E53 => { // tRNS
                for i in 0..chunk_length {
                    let (r,g,b,_) = palette[i].get();
                    let a = src[sp];
                    sp += 1;
                    palette[i].set(r,g,b,a);
                }
            },
            0x6348524D => { // cHRM
                //println!("cHRM {}",chunk_length);
                // chromaticity coordinates of display
                sp += chunk_length;
            },
            // dSIG (digital signature)
            0x65584966 => { // eXIf
                //println!("eXIf {}",chunk_length);
                // EXIF metadata
                sp += chunk_length;
            },
            0x67414D41 => { // gAMA
                let level = ((src[sp] as u32) << 24) | ((src[sp + 1] as u32) << 16) | ((src[sp + 2] as u32) << 8) | (src[sp + 3] as u32);
                gamma = (level as f32) / 100000.0;
                sp += chunk_length;
            },
            0x68495354 => { // hIST
                //println!("hIST {}",chunk_length);
                // histogram
                sp += chunk_length;
            },
            // iCCP (ICC color profile)
            0x69545874 => { // iTXt
                //println!("iTXt {}",chunk_length);
                // UTF-8 text
                sp += chunk_length;
            },
            0x70485973 => { // pHYs
                //println!("pHYs {}",chunk_length);
                // pixel aspect ratio
                sp += chunk_length;
            },
            0x73424954 => { // sBIT
                //println!("sBIT {}",chunk_length);
                // color accuracy
                sp += chunk_length;
            },
            0x73504C54 => { // sPLT
                //println!("sPLT {}",chunk_length);
                // palette in case colors are not available
                sp += chunk_length;
            },
            // sRGB (sRGB colorspace)
            // sTER (stereo)
            0x74455874 => { // tEXt
                //println!("tEXt {}",chunk_length);
                // text in ISO/IEC 8859-1
                sp += chunk_length;
            },
            0x74494D45 => { // tIME
                //println!("tIME {}",chunk_length);
                // time of last change to image
                sp += chunk_length;
            },
            0x7A545874 => { // zTXt
                //println!("zTXt {}",chunk_length);
                // compressed text
                sp += chunk_length;
            },
            _ => { // anything else just ignore
                //println!("unknown chunk: {:02X} {:02X} {:02X} {:02X}",chunk_type >> 24,(chunk_type >> 16) & 255,(chunk_type >> 8) & 255,chunk_type & 255);
                sp += chunk_length;
            },
        }
        sp += 4; // also skip the CRC
    }

    // sanity check the palette
    if need_plte && !plte_present {
        //println!("PNG: palette needed but not present");
        return None;
    }

    // sanity check the data
    if !idat_found || !iend_found {
        //println!("PNG: missing IDAT or IEND chunk");
        return None;
    }

    if interlace == 1 {
        let ax0: [u32; 7] = [0,4,0,2,0,1,0];
        let ay0: [u32; 7] = [0,0,4,0,2,0,1];
        let adx: [u32; 7] = [8,8,4,4,2,2,1];
        let ady: [u32; 7] = [8,8,8,4,4,2,2];
        let mut awidth: [u32; 7] = [0; 7];
        let mut aheight: [u32; 7] = [0; 7];
        let mut astride: [u32; 7] = [0; 7];
        let mut apresent: [bool; 7] = [false; 7];
        let mut adsize: [u32; 7] = [0; 7];
        let mut total_dsize = 0;
        //println!("size: {}x{}",width,height);
        for i in 0..7 {
            awidth[i] = (width + adx[i] - ax0[i] - 1) / adx[i];
            aheight[i] = (height + ady[i] - ay0[i] - 1) / ady[i];
            astride[i] = match itype {
                Type::L1 => { (awidth[i] + 7) / 8 },
                Type::C1 => { (awidth[i] + 7) / 8 },
                Type::L2 => { (awidth[i] + 3) / 4 },
                Type::C2 => { (awidth[i] + 3) / 4 },
                Type::L4 => { (awidth[i] + 1) / 2 },
                Type::C4 => { (awidth[i] + 1) / 2 },
                Type::L8 => { awidth[i] },
                Type::RGB8 => { awidth[i] * 3 },
                Type::C8 => { awidth[i] },
                Type::LA8 => { awidth[i] * 2 },
                Type::RGBA8 => { awidth[i] * 4 },
                Type::L16 => { awidth[i] * 2 },
                Type::RGB16 => { awidth[i] * 6 },
                Type::LA16 => { awidth[i] * 4 },
                Type::RGBA16 => { awidth[i] * 8 },
            };
            apresent[i] = (awidth[i] != 0) && (aheight[i] != 0);
            adsize[i] = if apresent[i] { (astride[i] + 1) * aheight[i] } else { 0 };
            total_dsize += adsize[i];
            //println!("{}: size {}x{}, offset {},{}, step {},{}",i,awidth[i],aheight[i],ax0[i],ay0[i],adx[i],ady[i]);
        }
        let filtered_data = match inflate(&zipped_data,total_dsize) {
            Some(data) => { data },
            None => {
                //println!("PNG: ZIP INFLATE failed");
                return None;
            },
        };
        let mut sp = 0usize;
        let mut result = Mat::<T>::new(Vec2::<usize> { x: width as usize,y: height as usize, });
        for i in 0..7 {
            if apresent[i] {
                let raw_data = unfilter(&filtered_data[sp..sp + adsize[i] as usize],aheight[i] as usize,astride[i] as usize,bpp);
                decode_pixels(&mut result,&raw_data,awidth[i] as usize,aheight[i] as usize,width as usize,ax0[i] as usize,ay0[i] as usize,adx[i] as usize,ady[i] as usize,itype,&palette,gamma);
                sp += adsize[i] as usize;
            }
        }
        Some(result)
    } else
    {
        //let after0 = Instant::now();
        
        let filtered_data = match inflate(&zipped_data,(stride + 1) * height) {
            Some(data) => { data },
            None => {
                //println!("PNG: ZIP INFLATE failed");
                return None;
            },
        };
        
        //let after_inflate = Instant::now();
        
        let raw_data = unfilter(&filtered_data,height as usize,stride as usize,bpp);
        
        //let after_unfilter = Instant::now();
        
        let mut result = Mat::new(Vec2::<usize> { x: width as usize,y: height as usize });
        decode_pixels(&mut result,&raw_data,width as usize,height as usize,width as usize,0,0,1,1,itype,&palette,gamma);
        
        //let after_decode = Instant::now();

        //let total_duration = after_decode.duration_since(after0);
        //let inflate_duration = after_inflate.duration_since(after0);
        //let inflate_percentage = (100.0 * inflate_duration.as_secs_f32()) / (total_duration.as_secs_f32());
        //let unfilter_duration = after_unfilter.duration_since(after_inflate);
        //let unfilter_percentage = (100.0 * unfilter_duration.as_secs_f32()) / (total_duration.as_secs_f32());
        //let decode_duration = after_decode.duration_since(after_unfilter);
        //let decode_percentage = (100.0 * decode_duration.as_secs_f32()) / (total_duration.as_secs_f32());

        //println!("inflate: {} us ({}%)",inflate_duration.as_micros(),inflate_percentage);
        //println!("unfilter: {} us ({}%)",unfilter_duration.as_micros(),unfilter_percentage);
        //println!("decode: {} us ({}%)",decode_duration.as_micros(),decode_percentage);
        //println!("------------------");
        //println!("total: {} us (100.0%)",total_duration.as_micros());

        Some(result)
    }
}

pub fn encode<T: Pixel>(_src: &Mat<T>) -> Option<Vec<u8>> {
    None
}


/*
78 5E = 01111000 01011110: CM=8 (deflate), CINFO=7 (32k window size), FLEVEL=fast, FDICT=no, FCHECK=1E
78 DA = 01111000 11011010: CM=8 (deflate), CINFO=7 (32k window size), FLEVEL=maximum, FDICT=no, FCHECK=1A
*/
