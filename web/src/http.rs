use std::fmt::Write;

/// HTTP request method.
///
/// # Examples
///
/// ```
/// use web::Method;
///
/// assert_eq!(Method::Get, Method::Get);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    /// GET — retrieve a resource.
    Get,
    /// HEAD — like GET but without the response body.
    Head,
    /// POST — submit data.
    Post,
    /// PUT — replace a resource.
    Put,
    /// DELETE — remove a resource.
    Delete,
    /// OPTIONS — describe communication options.
    Options,
}

impl Method {
    fn parse(s: &str) -> Option<Self> {
        match s {
            "GET" => Some(Self::Get),
            "HEAD" => Some(Self::Head),
            "POST" => Some(Self::Post),
            "PUT" => Some(Self::Put),
            "DELETE" => Some(Self::Delete),
            "OPTIONS" => Some(Self::Options),
            _ => None,
        }
    }
}

/// A parsed HTTP/1.1 request.
///
/// Produced by [`Request::parse`] from a raw byte buffer.
///
/// # Examples
///
/// ```
/// use web::{Request, Method};
/// use web::http::ParseResult;
///
/// let raw = b"GET /hello HTTP/1.1\r\nHost: localhost\r\n\r\n";
/// match Request::parse(raw) {
///     ParseResult::Ok(req, _consumed) => {
///         assert_eq!(req.method, Method::Get);
///         assert_eq!(req.path, "/hello");
///     }
///     _ => panic!("expected successful parse"),
/// }
/// ```
#[derive(Debug)]
pub struct Request {
    /// The HTTP method (GET, POST, etc.).
    pub method: Method,
    /// The request path (e.g. `"/index.html"`).
    pub path: String,
    /// Headers as `(lowercase-name, value)` pairs.
    pub headers: Vec<(String, String)>,
    /// The request body (empty for most GET requests).
    pub body: Vec<u8>,
}

/// Outcome of attempting to parse a request from a byte buffer.
pub enum ParseResult {
    /// Successfully parsed; contains the request and number of bytes consumed.
    Ok(Request, usize),
    /// Not enough data yet — keep reading.
    Incomplete,
    /// Malformed request.
    BadRequest,
}

impl Request {
    /// Try to parse a complete HTTP/1.1 request from `buf`.
    ///
    /// Returns [`ParseResult::Incomplete`] if the buffer does not yet contain a
    /// full set of headers (no `\r\n\r\n` found) or the indicated
    /// `Content-Length` body has not arrived.
    pub fn parse(buf: &[u8]) -> ParseResult {
        let header_end = match find_double_crlf(buf) {
            Some(pos) => pos,
            None => return ParseResult::Incomplete,
        };

        let header_str = match std::str::from_utf8(&buf[..header_end]) {
            Ok(s) => s,
            Err(_) => return ParseResult::BadRequest,
        };

        let mut lines = header_str.split("\r\n");

        let request_line = match lines.next() {
            Some(line) if !line.is_empty() => line,
            _ => return ParseResult::BadRequest,
        };

        let mut parts = request_line.split(' ');

        let method = match parts.next().and_then(Method::parse) {
            Some(m) => m,
            None => return ParseResult::BadRequest,
        };

        let path = match parts.next() {
            Some(p) => p.to_string(),
            None => return ParseResult::BadRequest,
        };

        let mut headers = Vec::new();
        let mut content_length = 0usize;
        for line in lines {
            if line.is_empty() {
                break;
            }
            if let Some((key, value)) = line.split_once(": ") {
                if key.eq_ignore_ascii_case("content-length") {
                    content_length = value.parse().unwrap_or(0);
                }
                headers.push((key.to_lowercase(), value.to_string()));
            }
        }

        let body_start = header_end + 4;
        let total = body_start + content_length;
        if buf.len() < total {
            return ParseResult::Incomplete;
        }
        let body = buf[body_start..total].to_vec();

        ParseResult::Ok(Request { method, path, headers, body }, total)
    }

    /// Look up a header value by name (case-insensitive).
    ///
    /// # Examples
    ///
    /// ```
    /// use web::Request;
    /// use web::http::ParseResult;
    ///
    /// let raw = b"GET / HTTP/1.1\r\nContent-Type: text/html\r\n\r\n";
    /// if let ParseResult::Ok(req, _) = Request::parse(raw) {
    ///     assert_eq!(req.header("content-type"), Some("text/html"));
    ///     assert_eq!(req.header("Content-Type"), Some("text/html"));
    ///     assert_eq!(req.header("x-missing"), None);
    /// }
    /// ```
    pub fn header(&self, name: &str) -> Option<&str> {
        let name_lower = name.to_lowercase();
        self.headers
            .iter()
            .find(|(k, _)| *k == name_lower)
            .map(|(_, v)| v.as_str())
    }
}

/// An HTTP response.
///
/// Built with a fluent API and serialized to bytes with [`to_bytes`](Response::to_bytes).
///
/// # Examples
///
/// ```
/// use web::Response;
///
/// let resp = Response::new(200)
///     .header("Content-Type", "text/plain")
///     .body_str("hello");
/// let bytes = resp.to_bytes();
/// let text = String::from_utf8(bytes).unwrap();
/// assert!(text.starts_with("HTTP/1.1 200 OK\r\n"));
/// assert!(text.ends_with("hello"));
/// ```
pub struct Response {
    /// HTTP status code.
    pub status: u16,
    /// Response headers as `(name, value)` pairs.
    pub headers: Vec<(&'static str, String)>,
    /// Response body.
    pub body: Vec<u8>,
}

impl Response {
    /// Create a response with the given status code and empty body.
    pub fn new(status: u16) -> Self {
        Response {
            status,
            headers: Vec::new(),
            body: Vec::new(),
        }
    }

    /// Append a header.
    pub fn header(mut self, key: &'static str, value: impl Into<String>) -> Self {
        self.headers.push((key, value.into()));
        self
    }

    /// Set the body from raw bytes.
    pub fn body(mut self, data: Vec<u8>) -> Self {
        self.body = data;
        self
    }

    /// Set the body from a string.
    pub fn body_str(self, text: &str) -> Self {
        self.body(text.as_bytes().to_vec())
    }

    /// Serialize to bytes suitable for writing to a TCP stream.
    pub fn to_bytes(&self) -> Vec<u8> {
        let reason = status_reason(self.status);
        let mut buf = format!("HTTP/1.1 {} {}\r\n", self.status, reason);
        let _ = write!(buf, "Content-Length: {}\r\n", self.body.len());
        for (key, value) in &self.headers {
            let _ = write!(buf, "{key}: {value}\r\n");
        }
        buf.push_str("\r\n");
        let mut bytes = buf.into_bytes();
        bytes.extend_from_slice(&self.body);
        bytes
    }
}

fn status_reason(code: u16) -> &'static str {
    match code {
        101 => "Switching Protocols",
        200 => "OK",
        304 => "Not Modified",
        400 => "Bad Request",
        404 => "Not Found",
        500 => "Internal Server Error",
        _ => "Unknown",
    }
}

fn find_double_crlf(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple_get() {
        let raw = b"GET /hello HTTP/1.1\r\nHost: localhost\r\n\r\n";
        match Request::parse(raw) {
            ParseResult::Ok(req, consumed) => {
                assert_eq!(req.method, Method::Get);
                assert_eq!(req.path, "/hello");
                assert_eq!(req.header("host"), Some("localhost"));
                assert!(req.body.is_empty());
                assert_eq!(consumed, raw.len());
            }
            _ => panic!("expected Ok"),
        }
    }

    #[test]
    fn parse_post_with_body() {
        let raw = b"POST /data HTTP/1.1\r\nContent-Length: 5\r\n\r\nhello";
        match Request::parse(raw) {
            ParseResult::Ok(req, consumed) => {
                assert_eq!(req.method, Method::Post);
                assert_eq!(req.body, b"hello");
                assert_eq!(consumed, raw.len());
            }
            _ => panic!("expected Ok"),
        }
    }

    #[test]
    fn parse_incomplete_headers() {
        let raw = b"GET / HTTP/1.1\r\nHost: local";
        assert!(matches!(Request::parse(raw), ParseResult::Incomplete));
    }

    #[test]
    fn parse_incomplete_body() {
        let raw = b"POST / HTTP/1.1\r\nContent-Length: 100\r\n\r\nshort";
        assert!(matches!(Request::parse(raw), ParseResult::Incomplete));
    }

    #[test]
    fn parse_bad_method() {
        let raw = b"FROBNICATE / HTTP/1.1\r\n\r\n";
        assert!(matches!(Request::parse(raw), ParseResult::BadRequest));
    }

    #[test]
    fn header_lookup_case_insensitive() {
        let raw = b"GET / HTTP/1.1\r\nX-Custom: value\r\n\r\n";
        if let ParseResult::Ok(req, _) = Request::parse(raw) {
            assert_eq!(req.header("x-custom"), Some("value"));
            assert_eq!(req.header("X-Custom"), Some("value"));
            assert_eq!(req.header("X-CUSTOM"), Some("value"));
        }
    }

    #[test]
    fn response_serialization() {
        let resp = Response::new(200)
            .header("Content-Type", "text/plain")
            .body_str("hi");
        let bytes = resp.to_bytes();
        let text = String::from_utf8(bytes).unwrap();
        assert!(text.starts_with("HTTP/1.1 200 OK\r\n"));
        assert!(text.contains("Content-Length: 2\r\n"));
        assert!(text.contains("Content-Type: text/plain\r\n"));
        assert!(text.ends_with("\r\n\r\nhi"));
    }

    #[test]
    fn response_empty_body() {
        let resp = Response::new(404);
        let bytes = resp.to_bytes();
        let text = String::from_utf8(bytes).unwrap();
        assert!(text.starts_with("HTTP/1.1 404 Not Found\r\n"));
        assert!(text.contains("Content-Length: 0\r\n"));
        assert!(text.ends_with("\r\n\r\n"));
    }

    #[test]
    fn two_requests_in_one_buffer() {
        let raw = b"GET /a HTTP/1.1\r\n\r\nGET /b HTTP/1.1\r\n\r\n";
        if let ParseResult::Ok(first, consumed) = Request::parse(raw) {
            assert_eq!(first.path, "/a");
            if let ParseResult::Ok(second, _) = Request::parse(&raw[consumed..]) {
                assert_eq!(second.path, "/b");
            } else {
                panic!("expected second request");
            }
        } else {
            panic!("expected first request");
        }
    }
}
