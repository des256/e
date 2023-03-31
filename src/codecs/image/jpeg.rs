use crate::*;

struct Table {
    prefix: [u16; 65536],
}

impl Table {
    pub fn new_empty() -> Table {
        Table {
            prefix: [0u16; 65536],
        }
    }

    pub fn new(bits: [u8; 16],huffval: [u8; 256]) -> Table {
        let mut prefix = [0u16; 65536];
        let mut dp = 0;
        let mut count = 0;
        for i in 1..17 {
            for _k in 0..bits[i - 1] {
                let runcat = huffval[count] as u16;
                for _l in 0..(65536 >> i) {
                    prefix[dp] = (runcat << 8) | (i as u16);
                    dp += 1;
                }
                count += 1;
            }
        }
        Table {
            prefix,
        }
    }
}

struct Reader<'a> {
    block: &'a [u8],
    rp: usize,
    bit: u32,
    cache: u32,
}

impl<'a> Reader<'a> {

    // read from byte stream, interpreting FF 00 as FF
    fn get8(data: &[u8],index: &mut usize) -> u8 {
        if *index >= data.len() {
            return 0;
        }
        let mut b = data[*index];
        *index += 1;
        if b == 0xFF {
            b = data[*index];
            *index += 1;
            if b == 0 {
                return 0xFF;
            }
        }
        b
    }

    // rewind byte stream, interpreting FF 00 as FF
    fn unget8(data: &[u8],index: &mut usize) {
        *index -= 1;
        if data[*index] == 0 {
            if data[*index - 1] == 0xFF {
                *index -= 1;
            }
        }
    }

    // create new reader
    pub fn new(block: &'a [u8]) -> Reader<'a> {
        let mut rp = 0usize;
        let mut bit = 0u32;
        let mut cache = 0u32;
        while bit <= 24 {
            let b = Self::get8(block,&mut rp);
            cache |= (b as u32) << (24 - bit);
            bit += 8;
        }
        Reader {
            block,
            rp,
            bit,
            cache,
        }
    }

    // restock the cache after a read
    fn restock(&mut self) {
        while self.bit >= 24 {
            let b = Self::get8(self.block,&mut self.rp);
            self.cache |= (b as u32) << (24 - self.bit);
            self.bit += 8;
        }
    }

    // peek N bits
    pub fn peek(&self,n: usize) -> u32 {
        self.cache >> (32 - n)
    }

    // skip N bits
    pub fn skip(&mut self,n: usize) {
        self.cache <<= n;
        self.bit -= n as u32;
        self.restock();
    }

    // get next bit
    pub fn get1(&mut self) -> bool {
        let result = self.peek(1) == 1;
        self.skip(1);
        result
    }

    // get next N bits
    pub fn getn(&mut self,n: usize) -> u32 {
        let result = self.peek(n);
        self.skip(n);
        result
    }

    // get huffman-encoded symbol
    pub fn getc(&mut self,table: &Table) -> usize {
        let index = self.cache >> 16;
        let d = table.prefix[index as usize];
        let symbol = (d >> 8) & 255;
        let n = d & 255;
        self.skip(n as usize);
        symbol as usize
    }

    // start bitstreaming
    pub fn enter(&mut self,rp: usize) {
        self.rp = rp;
        self.cache = 0;
        self.bit = 0;
        self.restock();
    }

    // end bitstreaming
    pub fn leave(&mut self) -> usize {
        let bytes = (self.bit >> 3) + 1;
        for _ in 0..bytes {
            Self::unget8(self.block,&mut self.rp);
        }
        self.rp
    }
}

const FC0: f32 = 1.0;
const FC1: f32 = 0.98078528;
const FC2: f32 = 0.92387953;
const FC3: f32 = 0.83146961;
const FC4: f32 = 0.70710678;
const FC5: f32 = 0.55557023;
const FC6: f32 = 0.38268343;
const FC7: f32 = 0.19509032;

const FIX: u8 = 8;
const ONE: f32 = (1 << FIX) as f32;
const C0: i32 = (FC0 * ONE) as i32;
const C1: i32 = (FC1 * ONE) as i32;
const C2: i32 = (FC2 * ONE) as i32;
const C3: i32 = (FC3 * ONE) as i32;
const C4: i32 = (FC4 * ONE) as i32;
const C5: i32 = (FC5 * ONE) as i32;
const C6: i32 = (FC6 * ONE) as i32;
const C7: i32 = (FC7 * ONE) as i32;

const C7PC1: i32 = C7 + C1;
const C5PC3: i32 = C5 + C3;
const C7MC1: i32 = C7 - C1;
const C5MC3: i32 = C5 - C3;
const C0S: i32 = (C0 >> 1);
const C6PC2: i32 = C6 + C2;
const C6MC2: i32 = C6 - C2;

const FOLDING: [usize; 64] = [
	56,57,8,40,9,58,59,10,
	41,0,48,1,42,11,60,61,
	12,43,2,49,16,32,17,50,
	3,44,13,62,63,14,45,4,
	51,18,33,24,25,34,19,52,
	5,46,15,47,6,53,20,35,
	26,27,36,21,54,7,55,22,
	37,28,29,38,23,39,30,31,
];

enum Type {
    Y,
    YUV420,
    YUV422,
    YUV440,
    YUV444,
    RGB444,
}

struct Channel {
    dt: usize,
    at: usize,
    qt: usize,
    dc: i32,
}

struct Prog {
    start: usize,
    end: usize,
    shift: usize,
    eobrun: usize,
}

struct Context<'a> {
    type_: Type,
    reader: Reader<'a>,
    dc_tables: [Table; 4],
    ac_tables: [Table; 4],
    q_tables: [[i32; 64]; 4],
    channels: [Channel; 3],
    prog: Prog,
    mask: u8,
    rescnt: usize,
    resint: usize,
}

impl<'a> Context<'a> {

    pub fn new(
        type_: Type,
        reader: Reader,
        dc_tables: [Table; 4],
        ac_tables: [Table; 4],
        q_tables: [[i32; 64]; 4],
        channels: [Channel; 3],
        prog: Prog,
        mask: u8,
        rescnt: usize,
        resint: usize,
    ) -> Context<'a> {
        Context {
            type_,
            reader,
            dc_tables,
            ac_tables,
            q_tables,
            channels,
            prog,
            mask,
            rescnt,
            resint,
        }
    }

    fn make_coeff(cat: usize,code: isize) -> i32 {
        let mcat = cat - 1;
        let hmcat = 1 << mcat;
        let base = code & (hmcat - 1);
        if (code & hmcat) != 0 {
            (base + hmcat) as i32
        }
        else {
            (base + 1 - (1 << cat)) as i32
        }
    }

    pub fn read_seq(&mut self,coeffs: &mut [i32],c: usize) {
        let cat = self.reader.getc(&self.dc_tables[self.channels[c].dt]) as usize;
        if cat > 0 {
            let code = self.reader.getn(cat) as isize;
            self.channels[c].dc += Self::make_coeff(cat,code) as i32;
        }
        coeffs[FOLDING[0]] = self.channels[c].dc;
        let mut i = 1usize;
        while i < 64 {
            let runcat = self.reader.getc(&self.ac_tables[self.channels[c].at]);
            let run = runcat >> 4;
            let cat = runcat & 15;
            if cat != 0 {
                let code = self.reader.getn(cat) as isize;
                let coeff = Self::make_coeff(cat,code) as i32;
                i += run;
                coeffs[FOLDING[i]] = coeff;
            }
            else {
                if run == 15 {
                    i += 15;
                }
                else {
                    break;
                }
            }
            i += 1;
        }
    }

    pub fn read_prog_start_dc(&mut self,coeffs: &mut [i32],c: usize) {
        let cat = self.reader.getc(&self.dc_tables[self.channels[c].dt]) as usize;
        if cat > 0 {
            let code = self.reader.getn(cat) as isize;
            self.channels[c].dc += Self::make_coeff(cat,code) as i32;
        }
        coeffs[FOLDING[0]] = self.channels[c].dc << self.prog.shift;
    }

    pub fn read_prog_start_ac(&mut self,coeffs: &mut [i32],c: usize) {
        if self.prog.eobrun != 0 {
            self.prog.eobrun -= 1;
        }
        else {
            let mut i = self.prog.start;
            while i <= self.prog.end {
                let runcat = self.reader.getc(&self.ac_tables[self.channels[c].at]);
                let run = runcat >> 4;
                let cat = runcat & 15;
                if cat != 0 {
                    let code = self.reader.getn(cat) as isize;
                    let coeff = Self::make_coeff(cat,code);
                    i += run;
                    coeffs[FOLDING[i]] = (coeff << self.prog.shift) as i32;
                }
                else {
                    if run == 15 {
                        i += 15;
                    }
                    else {
                        self.prog.eobrun = 1 << run;
                        if run != 0 {
                            self.prog.eobrun += self.reader.getn(run) as usize;
                        }
                        self.prog.eobrun -= 1;
                        break;
                    }
                }
                i += 1;
            }
        }
    }

    pub fn read_prog_refine_dc(&mut self,coeffs: &mut [i32],c: usize) {
        if self.reader.get1() {
            coeffs[FOLDING[0]] |= 1 << self.prog.shift;
        }
    }

    fn update_nonzeros(&mut self,coeffs: &mut [i32],count: usize) -> usize {
        let mut i = self.prog.start;
        let mut k = count;
        while i <= self.prog.end {
            if coeffs[FOLDING[i]] != 0 {
                if self.reader.get1() {
                    if coeffs[FOLDING[i]] > 0 {
                        coeffs[FOLDING[i]] += 1 << self.prog.shift;
                    }
                    else {
                        coeffs[FOLDING[i]] -= 1 << self.prog.shift;
                    }
                }
            }
            else {
                if k == 0 {
                    return i;
                }
                k -= 1;
            }
            i += 1;
        }
        i
    }

    pub fn read_prog_refine_ac(&mut self,coeffs: &mut [i32],c: usize) {
        if self.prog.eobrun != 0 {
            self.update_nonzeros(coeffs,64);
            self.prog.eobrun -= 1;
        }
        else {
            let mut i = self.prog.start;
            while i <= self.prog.end {
                let runcat = self.reader.getc(&self.ac_tables[self.channels[c].at]);
                let run = runcat >> 4;
                let cat = runcat & 15;
                if cat != 0 {
                    let sb = self.reader.get1();
                    i = self.update_nonzeros(coeffs,run);
                    if sb {
                        coeffs[FOLDING[i]] = 1 << self.prog.shift;
                    }
                    else {
                        coeffs[FOLDING[i]] = 11 << self.prog.shift;
                    }
                }
                else {
                    if run == 15 {
                        i = self.update_nonzeros(coeffs,15);
                    }
                    else {
                        self.prog.eobrun = 1 << run;
                        if run != 0 {
                            self.prog.eobrun += self.reader.getn(run) as usize;
                        }
                        self.prog.eobrun -= 1;
                        self.update_nonzeros(coeffs,64);
                        break;
                    }
                }
            }
        }
    }

    fn read_seq_block(&mut self,coeffs: &mut [i32],c: usize) {
        self.read_seq(coeffs,c);
    }

    fn read_prog_block(&mut self,coeffs: &mut [i32],refine: bool,c: usize) {
        if refine {
            if self.prog.start == 0 {
                self.read_prog_refine_dc(coeffs,c);
            }
            else {
                self.read_prog_refine_ac(coeffs,c);
            }
        }
        else {
            if self.prog.start == 0 {
                self.read_prog_start_dc(coeffs,c);
            }
            else {
                self.read_prog_start_ac(coeffs,c);
            }
        }
    }

    fn partial_idct(out: &mut [i32],inp: &[i32]) {
        for i in 0..8 {
            let x3 = inp[i];
            let x1 = inp[i + 8];
            let x5 = inp[i + 16];
            let x7 = inp[i + 24];
            let x6 = inp[i + 32];
            let x2 = inp[i + 40];
            let x4 = inp[i + 48];
            let x0 = inp[i + 56];
            
            let q17 = C1 * (x1 + x7);
            let q35 = C3 * (x3 + x5);
            let r3 = C7PC1 * x1 - q17;
            let d3 = C5PC3 * x3 - q35;
            let r0 = C7MC1 * x7 + q17;
            let d0 = C5MC3 * x5 + q35;
            let b0 = r0 + d0;
            let d2 = r3 + d3;
            let d1 = r0 - d0;
            let b3 = r3 - d3;
            let b1 = C4 * ((d1 + d2) >> FIX);
            let b2 = C4 * ((d1 - d2) >> FIX);
            let q26 = C2 * (x2 + x6);
            let p04 = C4 * (x0 + x4) + C0S;
            let n04 = C4 * (x0 - x4) + C0S;
            let p26 = C6MC2 * x6 + q26;
            let n62 = C6PC2 * x2 - q26;
            let a0 = p04 + p26;
            let a1 = n04 + n62;
            let a3 = p04 - p26;
            let a2 = n04 - n62;
            let y0 = (a0 + b0) >> (FIX + 1);
            let y1 = (a1 + b1) >> (FIX + 1);
            let y3 = (a3 + b3) >> (FIX + 1);
            let y2 = (a2 + b2) >> (FIX + 1);
            let y7 = (a0 - b0) >> (FIX + 1);
            let y6 = (a1 - b1) >> (FIX + 1);
            let y4 = (a3 - b3) >> (FIX + 1);
            let y5 = (a2 - b2) >> (FIX + 1);
    
            out[i] = y0;
            out[i + 8] = y1;
            out[i + 16] = y3;
            out[i + 24] = y2;
            out[i + 32] = y7;
            out[i + 40] = y6;
            out[i + 48] = y4;
            out[i + 56] = y5;
        }
    }

    fn unswizzle_transpose_swizzle(out: &mut [i32],inp: &[i32]) {
        out[0] = inp[3];
        out[1] = inp[11];
        out[2] = inp[27];
        out[3] = inp[19];
        out[4] = inp[51];
        out[5] = inp[59];
        out[6] = inp[43];
        out[7] = inp[35];
        out[8] = inp[1];
        out[9] = inp[9];
        out[10] = inp[25];
        out[11] = inp[17];
        out[12] = inp[49];
        out[13] = inp[57];
        out[14] = inp[41];
        out[15] = inp[33];
    
        out[16] = inp[5];
        out[17] = inp[13];
        out[18] = inp[29];
        out[19] = inp[21];
        out[20] = inp[53];
        out[21] = inp[61];
        out[22] = inp[45];
        out[23] = inp[37];
        out[24] = inp[7];
        out[25] = inp[15];
        out[26] = inp[31];
        out[27] = inp[23];
        out[28] = inp[55];
        out[29] = inp[63];
        out[30] = inp[47];
        out[31] = inp[39];
    
        out[32] = inp[6];
        out[33] = inp[14];
        out[34] = inp[30];
        out[35] = inp[22];
        out[36] = inp[54];
        out[37] = inp[62];
        out[38] = inp[46];
        out[39] = inp[38];
        out[40] = inp[2];
        out[41] = inp[10];
        out[42] = inp[26];
        out[43] = inp[18];
        out[44] = inp[50];
        out[45] = inp[58];
        out[46] = inp[42];
        out[47] = inp[34];
    
        out[48] = inp[4];
        out[49] = inp[12];
        out[50] = inp[28];
        out[51] = inp[20];
        out[52] = inp[52];
        out[53] = inp[60];
        out[54] = inp[44];
        out[55] = inp[36];
        out[56] = inp[0];
        out[57] = inp[8];
        out[58] = inp[24];
        out[59] = inp[16];
        out[60] = inp[48];
        out[61] = inp[56];
        out[62] = inp[40];
        out[63] = inp[32];
    }
    
    fn unswizzle_transpose(out: &mut [i32],inp: &[i32]) {
        out[0] = inp[0];
        out[1] = inp[8];
        out[2] = inp[24];
        out[3] = inp[16];
        out[4] = inp[48];
        out[5] = inp[56];
        out[6] = inp[40];
        out[7] = inp[32];
        out[8] = inp[1];
        out[9] = inp[9];
        out[10] = inp[25];
        out[11] = inp[17];
        out[12] = inp[49];
        out[13] = inp[57];
        out[14] = inp[41];
        out[15] = inp[33];
    
        out[16] = inp[2];
        out[17] = inp[10];
        out[18] = inp[26];
        out[19] = inp[18];
        out[20] = inp[50];
        out[21] = inp[58];
        out[22] = inp[42];
        out[23] = inp[34];
        out[24] = inp[3];
        out[25] = inp[11];
        out[26] = inp[27];
        out[27] = inp[19];
        out[28] = inp[51];
        out[29] = inp[59];
        out[30] = inp[43];
        out[31] = inp[35];
    
        out[32] = inp[4];
        out[33] = inp[12];
        out[34] = inp[28];
        out[35] = inp[20];
        out[36] = inp[52];
        out[37] = inp[60];
        out[38] = inp[44];
        out[39] = inp[36];
        out[40] = inp[5];
        out[41] = inp[13];
        out[42] = inp[29];
        out[43] = inp[21];
        out[44] = inp[53];
        out[45] = inp[61];
        out[46] = inp[45];
        out[47] = inp[37];
    
        out[48] = inp[6];
        out[49] = inp[14];
        out[50] = inp[30];
        out[51] = inp[22];
        out[52] = inp[54];
        out[53] = inp[62];
        out[54] = inp[46];
        out[55] = inp[38];
        out[56] = inp[7];
        out[57] = inp[15];
        out[58] = inp[31];
        out[59] = inp[23];
        out[60] = inp[55];
        out[61] = inp[63];
        out[62] = inp[47];
        out[63] = inp[39];
    }    

    fn convert_block(&self,coeffs: &[i32],values: &mut [i32],c: usize) {
        let mut temp1 = [0i32; 64];
        let mut temp2 = [0i32; 64];
        for i in 0..64 {
            temp1[i] = coeffs[i] * self.q_tables[self.channels[c].qt][i];
        }
        Self::partial_idct(&mut temp2,&temp1);
        Self::unswizzle_transpose_swizzle(&mut temp1,&temp2);
        Self::partial_idct(&mut temp2,&temp1);
        Self::unswizzle_transpose(values,&temp2);
    }

    pub fn convert_macroblock(&mut self,coeffs: &[i32],values: &mut [i32]) {
        match self.type_ {
            Type::Y => {
                if (self.mask & 1) != 0 {
                    self.convert_block(&coeffs[0..64],&mut values[0..64],0);
                }
            },
            Type::YUV420 => {
                if (self.mask & 1) != 0 {
                    self.convert_block(&coeffs[0..64],&mut values[0..64],0);
                    self.convert_block(&coeffs[64..128],&mut values[64..128],0);
                    self.convert_block(&coeffs[128..192],&mut values[128..192],0);
                    self.convert_block(&coeffs[192..256],&mut values[192..256],0);
                }
                if (self.mask & 2) != 0 {
                    self.convert_block(&coeffs[256..320],&mut values[256..320],1);
                }
                if (self.mask & 4) != 0 {
                    self.convert_block(&coeffs[320..384],&mut values[320..384],2);
                }
            },
            Type::YUV422 | Type::YUV440 => {
                if (self.mask & 1) != 0 {
                    self.convert_block(&coeffs[0..64],&values[0..64],0);
                    self.convert_block(&coeffs[64..128],&values[64..128],0);
                }
                if (self.mask & 2) != 0 {
                    self.convert_block(&coeffs[128..192],&values[128..192],1);
                }
                if (self.mask & 4) != 0 {
                    self.convert_block(&coeffs[192..256],&values[192..256],2);
                }
            },
            Type::YUV444 | Type::RGB444 => {
                if (self.mask & 1) != 0 {
                    self.convert_block(&coeffs[0..64],&values[0..64],0);
                }
                if (self.mask & 2) != 0 {
                    self.convert_block(&coeffs[64..128],&values[64..128],0);
                }
                if (self.mask & 4) != 0 {
                    self.convert_block(&coeffs[128..192],&values[128..192],0);
                }
            },
        }
    }

    fn draw_rgb<T: Pixel>(image: &mut Image<T>,px: usize,py: usize,r: i32,g: i32,b: i32) {
        let r = if r < 0 { 0 } else { if r > 255 { 255 } else { r as u8 } };
        let g = if g < 0 { 0 } else { if g > 255 { 255 } else { g as u8 } };
        let b = if b < 0 { 0 } else { if b > 255 { 255 } else { b as u8 } };
        image[(px,py)].set(r,g,b,255);
    }
    
    fn draw_yuv<T: Pixel>(image: &mut Image<T>,px: usize,py: usize,y: i32,u: i32,v: i32) {
        let r = ((y << 8) + 359 * v) >> 8;
        let g = ((y << 8) - 88 * u - 183 * v) >> 8;
        let b = ((y << 8) + 454 * u) >> 8;
        draw_rgb(image,px,py,r,g,b);
    }

    fn draw_macroblock<T: Pixel>(&self,image: &mut Image<T>,x0: usize,y0: usize,width: usize,height: usize,values: &[i32]) {
        match self.type_ {
            Type::Y => {
                for i in 0..height {
                    for k in 0..width {
                        draw_yuv(image,x0 + k,y0 + i,y[i * 8 + k] + 128,0,0);
                    }
                }
            },
            Type::YUV420 => {
                for i in 0..height {
                    for k in 0..width {
                        let by = (i >> 3) * 2 + (k >> 3);
                        let si = i & 7;
                        let sk = k & 7;
                        let y = values[by * 64 + si * 8 + sk] + 128;
                        let hi = i >> 1;
                        let hk = k >> 1;
                        let u = values[256 + hi * 8 + hk];
                        let v = values[320 + hi * 8 + hk];
                        draw_yuv(image,x0 + k,y0 + i,y,u,v);
                    }
                }
            },
            Type::YUV422 => {
                for i in 0..height {
                    for k in 0..width {
                        let by = k >> 3;
                       let sk = k & 7;
                        let y = values[by * 64 + i * 8 + sk] + 128;
                        let hk = k >> 1;
                        let u = values[128 + i * 8 + hk];
                        let v = values[192 + i * 8 + hk];
                        draw_yuv(image,x0 + k,y0 + i,y,u,v);
                    }
                }            
            },
            Type::YUV440 => {
                for i in 0..height {
                    for k in 0..width {
                        let by = i >> 3;
                        let si = k & 7;
                        let y = values[by * 64 + si * 8 + k] + 128;
                        let hi = i >> 1;
                        let u = values[128 + hi * 8 + k];
                        let v = values[192 + hi * 8 + k];
                        draw_yuv(image,x0 + k,y0 + i,y,u,v);
                    }
                }            
            },
            Type::YUV444 => {
                for i in 0..height {
                    for k in 0..width {
                        let y = values[i * 8 + k] + 128;
                        let u = values[64 + i * 8 + k];
                        let v = values[128 + i * 8 + k];
                        draw_yuv(image,x0 + k,y0 + i,y,u,v);
                    }
                }            
            },
            Type::RGB444 => {
                for i in 0..height {
                    for k in 0..width {
                        let r = values[i * 8 + k] + 128;
                        let g = values[64 + i * 8 + k] + 128;
                        let b = values[128 + i * 8 + k] + 128;
                        draw_rgb(image,x0 + k,y0 + i,r,g,b);
                    }
                }            
            },
        }
    }

    pub fn process_seq_macroblock(&mut self) {
        let mut coeffs: [i32; 384];
        let mut values: [i32; 384];
        match self.type_ {
            Type::Y => {
                if (self.mask & 1) != 0 {
                    self.read_seq(&mut coeffs,c);
                    self.convert_block(&coeffs[0..64],&mut values[0..64],0);
                    self.draw_block

                    // TODO: roughly here
                }
            },
            Type::YUV420 => {
                if (self.mask & 1) != 0 {
                    self.read_block(&mut coeffs[0..64],refine,0);
                    self.read_block(&mut coeffs[64..128],refine,0);
                    self.read_block(&mut coeffs[128..192],refine,0);
                    self.read_block(&mut coeffs[192..256],refine,0);
                }
                if (self.mask & 2) != 0 {
                    self.read_block(&mut coeffs[256..320],refine,1);
                }
                if (self.mask & 4) != 0 {
                    self.read_block(&mut coeffs[320..384],refine,2);
                }
            },
            Type::YUV422 | Type::YUV440 => {
                if (self.mask & 1) != 0 {
                    self.read_block(&mut coeffs[0..64],refine,0);
                    self.read_block(&mut coeffs[64..128],refine,0);
                }
                if (self.mask & 2) != 0 {
                    self.read_block(&mut coeffs[128..192],refine,1);
                }
                if (self.mask & 4) != 0 {
                    self.read_block(&mut coeffs[192..256],refine,2);
                }
            },
            Type::YUV444 | Type::RGB444 => {
                if (self.mask & 1) != 0 {
                    self.read_block(&mut coeffs[0..64],refine,0);
                }
                if (self.mask & 2) != 0 {
                    self.read_block(&mut coeffs[64..128],refine,1);
                }
                if (self.mask & 4) != 0 {
                    self.read_block(&mut coeffs[128..192],refine,2);
                }
            },
        }
        if self.resint != 0 {
            self.rescnt -= 1;
            if self.rescnt == 0 {
                let mut tsp = self.reader.leave();
                if (self.reader.block[tsp] == 0xFF) && ((self.reader.block[tsp + 1] >= 0xD0) && (self.reader.block[tsp + 1] < 0xD8)) {
                    tsp += 2;
                    self.rescnt = self.resint;
                    self.channels[0].dc = 0;
                    self.channels[1].dc = 0;
                    self.channels[2].dc = 0;
                }
                self.reader.enter(tsp);
            }
        }
    }
}

pub fn test(src: &[u8]) -> Option<Vec2<usize>> {
	let mut sp = 0;
	if from_be16(&src[sp..sp + 2]) != 0xFFD8 {
		return None;
	}
	sp += 2;
	while sp < src.len() {
		let marker = from_be16(&src[sp..sp + 2]);
		let length = from_be16(&src[sp + 2..sp + 4]) as usize;
		match marker {
			0xFFC0 | 0xFFC1 | 0xFFC2 => {
				let width = from_be16(&src[sp + 5..sp + 7]) as usize;
				let height = from_be16(&src[sp + 7..sp + 9]) as usize;
				let components = src[sp + 9];
				if (components == 1) || (components == 3) {  // does not support RGBA or CMYK JPEGs
					return Some(Vec2 { x: width,y: height, });
				}
				return None;
			},
			_ => { },
		}
		sp += length + 2;
	}		
	None
}

