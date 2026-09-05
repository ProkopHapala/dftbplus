//! graphene-build — CLI for generating graphene nanostructure geometries.
//!
//! Builds ribbons, sheets, flakes, and PAHs and writes them to XYZ files
//! (standard or extended XYZ with lattice vectors for periodic systems).
//!
//! Usage:
//!   graphene-build zigzag  --width 4 --length 10 --passivate --out ribbon.xyz
//!   graphene-build armchair --width 6 --length 8 --passivate --out arm.xyz
//!   graphene-build sheet   --nx 5 --ny 5 --periodic --out sheet.xyz
//!   graphene-build flake   --radius 10 --shape circle --passivate --out flake.xyz
//!   graphene-build pah     --shells 2 --out coronene.xyz
//!
//! Run with --help for full options.

use std::process::ExitCode;
use std::f64::consts;
use rust_dftb::geometry::*;

#[derive(Debug)]
struct Args {
    kind: String,
    width: usize,
    length: usize,
    nx: usize,
    ny: usize,
    radius: f64,
    shells: usize,
    shape: FlakeShape,
    passivate: bool,
    periodic: bool,
    acc: f64,
    out: String,
    recenter: bool,
    info: bool,
}

fn print_help() {
    eprintln!(
"graphene-build — generate graphene nanostructure geometries

USAGE:
    graphene-build <KIND> [OPTIONS] --out <FILE.xyz>

KINDS:
    zigzag      Zigzag-edge graphene ribbon (edges along x)
    armchair    Armchair-edge graphene ribbon (edges along x)
    sheet       Rectangular graphene sheet (supercell)
    flake       Circular or hexagonal graphene flake
    pah         Hexagonal polycyclic aromatic hydrocarbon (benzene, coronene, ...)

OPTIONS:
    --width <N>     Ribbon width in atom rows      [default: 4]
    --length <N>    Ribbon length in unit cells    [default: 10]
    --nx <N>        Sheet size in a1 direction      [default: 5]
    --ny <N>        Sheet size in a2 direction      [default: 5]
    --radius <R>    Flake radius in Angstrom        [default: 10.0]
    --shells <N>    PAH ring shells (0=benzene)     [default: 2]
    --shape <S>     Flake shape: circle or hex      [default: circle]
    --passivate     Add H atoms to edge carbons
    --periodic      Periodic boundary conditions (sheet/ribbon along x)
    --acc <A>       C-C bond length in Angstrom     [default: 1.42]
    --recenter      Shift centroid to origin
    --info          Print structure info to stderr
    --out <FILE>    Output XYZ file path            [required]
    -h, --help      Show this help

EXAMPLES:
    graphene-build zigzag --width 4 --length 20 --passivate --out z4x20.xyz
    graphene-build armchair --width 6 --length 10 --passivate --out a6x10.xyz
    graphene-build sheet --nx 10 --ny 10 --periodic --out sheet10x10.xyz
    graphene-build flake --radius 15 --shape hex --passivate --out flake.xyz
    graphene-build pah --shells 3 --out pah3.xyz
");
}

fn parse_args(argv: &[String]) -> Result<Args, String> {
    let mut args = Args {
        kind: String::new(),
        width: 4,
        length: 10,
        nx: 5,
        ny: 5,
        radius: 10.0,
        shells: 2,
        shape: FlakeShape::Circle,
        passivate: false,
        periodic: false,
        acc: A_CC,
        out: String::new(),
        recenter: false,
        info: false,
    };

    let mut positional: Vec<String> = Vec::new();
    let mut i = 0;
    while i < argv.len() {
        let a = &argv[i];
        match a.as_str() {
            "-h" | "--help" => { print_help(); std::process::exit(0); }
            "--width" => { i += 1; args.width = argv.get(i).ok_or("--width needs value")?.parse().map_err(|e: std::num::ParseIntError| format!("--width: {e}"))?; }
            "--length" => { i += 1; args.length = argv.get(i).ok_or("--length needs value")?.parse().map_err(|e: std::num::ParseIntError| format!("--length: {e}"))?; }
            "--nx" => { i += 1; args.nx = argv.get(i).ok_or("--nx needs value")?.parse().map_err(|e: std::num::ParseIntError| format!("--nx: {e}"))?; }
            "--ny" => { i += 1; args.ny = argv.get(i).ok_or("--ny needs value")?.parse().map_err(|e: std::num::ParseIntError| format!("--ny: {e}"))?; }
            "--radius" => { i += 1; args.radius = argv.get(i).ok_or("--radius needs value")?.parse().map_err(|e: std::num::ParseFloatError| format!("--radius: {e}"))?; }
            "--shells" => { i += 1; args.shells = argv.get(i).ok_or("--shells needs value")?.parse().map_err(|e: std::num::ParseIntError| format!("--shells: {e}"))?; }
            "--shape" => {
                i += 1;
                let s = argv.get(i).ok_or("--shape needs value")?;
                args.shape = match s.as_str() {
                    "circle" => FlakeShape::Circle,
                    "hex" => FlakeShape::Hex,
                    _ => return Err(format!("--shape: unknown '{s}' (use circle or hex)")),
                };
            }
            "--acc" => { i += 1; args.acc = argv.get(i).ok_or("--acc needs value")?.parse().map_err(|e: std::num::ParseFloatError| format!("--acc: {e}"))?; }
            "--out" => { i += 1; args.out = argv.get(i).ok_or("--out needs value")?.clone(); }
            "--passivate" => args.passivate = true,
            "--periodic" => args.periodic = true,
            "--recenter" => args.recenter = true,
            "--info" => args.info = true,
            _ => positional.push(a.clone()),
        }
        i += 1;
    }

    if positional.is_empty() {
        return Err("missing KIND (zigzag|armchair|sheet|flake|pah)".into());
    }
    args.kind = positional[0].clone();

    let valid = ["zigzag", "armchair", "sheet", "flake", "pah"];
    if !valid.contains(&args.kind.as_str()) {
        return Err(format!("unknown KIND '{}' (use: {})", args.kind, valid.join(", ")));
    }

    if args.out.is_empty() {
        return Err("--out <FILE> is required".into());
    }

    Ok(args)
}

fn build_structure(args: &Args) -> Result<(NanoStructure, Option<[[f64; 3]; 3]>), String> {
    let mut st = match args.kind.as_str() {
        "zigzag" => build_zigzag_ribbon(args.width, args.length, args.passivate, args.periodic, args.acc),
        "armchair" => build_armchair_ribbon(args.width, args.length, args.passivate, args.periodic, args.acc),
        "sheet" => build_sheet(args.nx, args.ny, args.periodic, args.acc),
        "flake" => build_flake(args.radius, args.shape, args.passivate, args.acc),
        "pah" => build_pah(args.shells, args.acc),
        _ => unreachable!(),
    };

    if args.recenter {
        st.recenter();
    }

    // For periodic sheet, compute lattice vectors
    let lattice = if args.periodic {
        match args.kind.as_str() {
            "sheet" => {
                let [a1, a2] = lattice_vectors(args.acc);
                Some([
                    [args.nx as f64 * a1[0], args.nx as f64 * a1[1], 0.0],
                    [args.ny as f64 * a2[0], args.ny as f64 * a2[1], 0.0],
                    [0.0, 0.0, 20.0], // vacuum in z
                ])
            }
            "zigzag" | "armchair" => {
                // Ribbon periodic along x: cell = (length * x_period, 0, 0)
                let xa = args.acc * consts::FRAC_PI_6.cos();
                let x_period = 2.0 * xa;
                let (lo, hi) = st.bbox();
                Some([
                    [args.length as f64 * x_period, 0.0, 0.0],
                    [0.0, (hi[1] - lo[1]) + 10.0, 0.0], // y extent + vacuum
                    [0.0, 0.0, 20.0],
                ])
            }
            _ => None,
        }
    } else {
        None
    };

    Ok((st, lattice))
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let args = match parse_args(&argv) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            eprintln!("run with --help for usage");
            return ExitCode::from(2);
        }
    };

    eprintln!("building {} ...", args.kind);
    let (mut st, lattice) = match build_structure(&args) {
        Ok((st, lat)) => (st, lat),
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(1);
        }
    };

    let n_c = st.count(Element::C);
    let n_h = st.count(Element::H);
    let (lo, hi) = st.bbox();
    let n_edge = st.find_edge_carbons().len();

    eprintln!(
        "  atoms: {} (C={}, H={}), bonds: {}, edge C: {}",
        st.natom(), n_c, n_h, st.nbond(), n_edge
    );
    eprintln!(
        "  bbox: x=[{:.2}, {:.2}]  y=[{:.2}, {:.2}]  z=[{:.2}, {:.2}]",
        lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]
    );
    if let Some(lat) = &lattice {
        eprintln!(
            "  lattice: a1=({:.3}, {:.3}, {:.3})  a2=({:.3}, {:.3}, {:.3})  a3=({:.3}, {:.3}, {:.3})",
            lat[0][0], lat[0][1], lat[0][2],
            lat[1][0], lat[1][1], lat[1][2],
            lat[2][0], lat[2][1], lat[2][2]
        );
    }

    let xyz = if lattice.is_some() {
        st.to_xyz_periodic(lattice.as_ref())
    } else {
        st.to_xyz()
    };

    match std::fs::write(&args.out, &xyz) {
        Ok(_) => eprintln!("  written to {}", args.out),
        Err(e) => {
            eprintln!("error writing {}: {e}", args.out);
            return ExitCode::from(1);
        }
    }

    if args.info {
        // Print full atom list to stderr
        eprintln!("\n--- atom list ---");
        for i in 0..st.natom() {
            let [x, y, z] = st.positions[i];
            eprintln!("  [{i:4}] {}  {x:10.4}  {y:10.4}  {z:10.4}", st.elements[i].symbol());
        }
        eprintln!("--- bonds ---");
        for (b, &[i, j]) in st.bonds.iter().enumerate() {
            eprintln!("  [{b:4}] {i:4} -- {j:4}");
        }
    }

    let _ = &mut st; // suppress unused_mut
    eprintln!("done.");
    ExitCode::SUCCESS
}
