//! Static file serving with MIME type detection and cache headers.

use crate::*;
use std::path::Path;

/// Serve a static file from `dir` for the given request path.
///
/// Maps `"/"` to `"index.html"`. Rejects paths containing `..` with a 400
/// response. Returns 404 if the file does not exist.
///
/// # Examples
///
/// ```no_run
/// use web::*;
///
/// let response = serve::static_file("/var/www", "/style.css");
/// assert_eq!(response.status, 200);
/// ```
pub fn static_file(dir: &str, request_path: &str) -> Response {
    let req_path = request_path.trim_start_matches('/');
    let req_path = if req_path.is_empty() {
        "index.html"
    } else {
        req_path
    };

    if req_path.contains("..") {
        return Response::new(400).body_str("Bad Request");
    }

    let file_path = Path::new(dir).join(req_path);

    let data = match std::fs::read(&file_path) {
        Ok(data) => data,
        Err(_) => return Response::new(404).body_str("Not Found"),
    };

    let mime = mime_for_path(&file_path);
    let cache = cache_policy(&file_path);

    Response::new(200)
        .header("Content-Type", mime)
        .header("Cache-Control", cache)
        .body(data)
}

fn mime_for_path(path: &Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()) {
        Some("html") => "text/html; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("js") => "application/javascript; charset=utf-8",
        Some("wasm") => "application/wasm",
        Some("json") => "application/json",
        Some("png") => "image/png",
        Some("jpg") | Some("jpeg") => "image/jpeg",
        Some("svg") => "image/svg+xml",
        Some("ico") => "image/x-icon",
        Some("txt") => "text/plain; charset=utf-8",
        _ => "application/octet-stream",
    }
}

fn cache_policy(path: &Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()) {
        Some("wasm") | Some("js") => "public, max-age=31536000, immutable",
        Some("html") => "no-cache",
        _ => "public, max-age=3600",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serves_existing_file() {
        let dir = std::env::temp_dir().join("web_test_serve");
        let _ = std::fs::create_dir_all(&dir);
        std::fs::write(dir.join("hello.txt"), "hi").unwrap();

        let resp = static_file(dir.to_str().unwrap(), "/hello.txt");
        assert_eq!(resp.status, 200);
        assert_eq!(resp.body, b"hi");
        assert!(resp.headers.iter().any(|(k, v)| *k == "Content-Type" && v.contains("text/plain")));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn returns_404_for_missing_file() {
        let resp = static_file("/tmp/web_test_nonexistent", "/nope.txt");
        assert_eq!(resp.status, 404);
    }

    #[test]
    fn root_maps_to_index_html() {
        let dir = std::env::temp_dir().join("web_test_index");
        let _ = std::fs::create_dir_all(&dir);
        std::fs::write(dir.join("index.html"), "<h1>hi</h1>").unwrap();

        let resp = static_file(dir.to_str().unwrap(), "/");
        assert_eq!(resp.status, 200);
        assert_eq!(resp.body, b"<h1>hi</h1>");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_path_traversal() {
        let resp = static_file("/tmp", "/../etc/passwd");
        assert_eq!(resp.status, 400);
    }

    #[test]
    fn correct_mime_types() {
        let dir = std::env::temp_dir().join("web_test_mime");
        let _ = std::fs::create_dir_all(&dir);

        for (name, expected_mime) in [
            ("a.html", "text/html"),
            ("b.css", "text/css"),
            ("c.js", "application/javascript"),
            ("d.wasm", "application/wasm"),
            ("e.json", "application/json"),
            ("f.png", "image/png"),
        ] {
            std::fs::write(dir.join(name), "x").unwrap();
            let resp = static_file(dir.to_str().unwrap(), &format!("/{name}"));
            assert_eq!(resp.status, 200, "failed for {name}");
            let ct = resp.headers.iter().find(|(k, _)| *k == "Content-Type").unwrap();
            assert!(ct.1.contains(expected_mime), "wrong mime for {name}: {}", ct.1);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }
}
