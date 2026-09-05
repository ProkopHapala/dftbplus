fn main() {
    // Link to system OpenBLAS for LAPACK eigensolver (dsyevd etc.)
    println!("cargo:rustc-link-lib=dylib=openblas");
    println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu/openblas-pthread/");
}
