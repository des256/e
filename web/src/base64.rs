const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Base64-encode `data` (with `=` padding).
pub(crate) fn encode(data: &[u8]) -> String {
    let mut out = String::with_capacity((data.len() + 2) / 3 * 4);

    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;

        out.push(ALPHABET[((triple >> 18) & 0x3F) as usize] as char);
        out.push(ALPHABET[((triple >> 12) & 0x3F) as usize] as char);

        if chunk.len() > 1 {
            out.push(ALPHABET[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }

        if chunk.len() > 2 {
            out.push(ALPHABET[(triple & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty() {
        assert_eq!(encode(b""), "");
    }

    #[test]
    fn one_byte() {
        assert_eq!(encode(b"f"), "Zg==");
    }

    #[test]
    fn two_bytes() {
        assert_eq!(encode(b"fo"), "Zm8=");
    }

    #[test]
    fn three_bytes() {
        assert_eq!(encode(b"foo"), "Zm9v");
    }

    #[test]
    fn padding_variations() {
        assert_eq!(encode(b"foobar"), "Zm9vYmFy");
        assert_eq!(encode(b"fooba"), "Zm9vYmE=");
        assert_eq!(encode(b"foob"), "Zm9vYg==");
    }

    #[test]
    fn hello_world() {
        assert_eq!(encode(b"Hello, World!"), "SGVsbG8sIFdvcmxkIQ==");
    }
}
