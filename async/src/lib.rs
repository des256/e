mod runtime;
pub use runtime::*;

pub trait AsyncRead {
    fn poll_read(self: Pin<&mut Self>,context: &mut Context<'_>,buffer: &mut ReadBuf<'_>) -> Poll<Result<()>>;
}

pub trait AsyncWrite {
    fn poll_write(self: Pin<&mut Self> context: &mut Context<'_>,buffer: &[u8]) -> Poll<Result<usize,Error>>;
    fn poll_flush(self: Pin<&mut Self>,context: &mut Context<'_>) -> Poll<Result<(),Error>>;
    fn poll_shutdown(self: Pin<&mut Self>,context: &mut Context<'_>) -> Poll<Result<(),Error>>;
    fn poll_write_vectored(self: Pin<&mut Self>,context: &mut Context<'_>,buffers: &[IoSlice<'_>]) -> Poll<Result<usize,Error>>;
}

pub trait AsyncSeek {
    fn start_seek(self: Pin<&mut Self>,position: SeekFrom) -> Result<()>;
    fn poll_complete(self: Pin<&mut Self>,context: &mut Context<'_>) -> Poll<Result<u64>>;
}

pub trait AsyncBufRead: AsyncRead {
    fn poll_fill_buf(self: Pin<&mut Self>,context: &mut Context<'_>) -> Poll<Result<&[u8]>>;
    fn consume(self: Pin<&mut Self>,bytes: usize);
}

pub struct BufReader<R> { }

pub struct BufWriter<W> { }

pub struct BufStream<RW> { }

// broadcast
// mpsc
// oneshot
// watch
// Barrier
// Mutex
// OnceCell
// RwLock
// Semaphore
// TcpListener
// TcpStream
// UdpSocket
// UnixListener
// UnixStream
// UnixDatagram
// unix::pipe
// windows::named_pipe
// File
// DirBuilder
// DirEntry
// OpenOptions
// ReadDir
