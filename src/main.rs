mod eval;
mod lfsr;

use std::collections::{HashMap, HashSet};

use aes::{
    cipher::{Block, BlockDecrypt, BlockEncrypt, Key, KeyInit},
    Aes128,
};
use clap::Parser;
use des::Des;
use eval::Expr;
use lfsr::{read_binary_string, solve_for_coefs, to_binary_string, Lfsr};
use sha2::Digest;

fn modulus(a: i128, m: i128) -> i128 {
    let r = a % m;
    if r < 0 {
        r + m
    } else {
        r
    }
}

pub fn modinv(a: i128, m: i128) -> i128 {
    let a = modulus(a, m);
    let gcd = gcd(a, m);
    if gcd != 1 {
        return a;
        // panic!("{} and {} are not coprime", a, m);
    }
    for i in 1..m {
        if (a * i) % m == 1 {
            return i;
        }
    }
    panic!("No modular inverse found");
}

fn moddiv(a: i128, b: i128, m: i128) -> i128 {
    modulus(modulus(a, m) * modinv(modulus(b, m), m), m)
}

fn modpow(mut base: i128, mut exp: i128, modulus: i128) -> i128 {
    let mut result = 1;
    base %= modulus;
    while exp > 0 {
        if exp % 2 == 1 {
            result = (result * base) % modulus;
        }
        exp >>= 1;
        base = (base * base) % modulus;
    }
    result
}

fn is_generator(g: i128, p: i128) -> bool {
    let mut nums = HashSet::new();
    for i in 1..p {
        let n = modpow(g, i, p);
        if nums.contains(&n) {
            return false;
        }
        nums.insert(n);
    }
    true
}

fn prime_factors(p: i128) -> HashMap<i128, i128> {
    let mut p = p;
    let mut factors = HashMap::new();
    let mut i = 2;
    while i <= p {
        if p % i == 0 {
            *factors.entry(i).or_insert(0) += 1;
            p /= i;
        } else {
            i += 1;
        }
    }
    if p > 1 {
        *factors.entry(p).or_insert(0) += 1;
    }
    factors
}

fn gcd(a: i128, b: i128) -> i128 {
    let mut a = a;
    let mut b = b;
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

fn extended_euclid(a: i128, b: i128) -> (i64, i64, i64) {
    let mut old_r = a as i64;
    let mut r = b as i64;
    let mut old_s = 1;
    let mut s = 0;
    let mut old_t = 0;
    let mut t = 1;

    while r != 0 {
        let os = s;
        let ot = t;

        let ri = old_r % r;
        let q = (old_r - ri) / r;
        old_r = r;
        r = ri;

        s = old_s - q * s;
        t = old_t - q * t;
        old_s = os;
        old_t = ot;
    }

    (old_r, old_s, old_t)
}

fn rsa_sig(n: i128, d: i128, m: i128) -> i128 {
    modpow(m, d, n)
}

fn rsa_ver(n: i128, e: i128, m: i128, s: i128) -> bool {
    modpow(s, e, n) == m
}

fn rsa_enc(n: i128, e: i128, m: i128) -> i128 {
    modpow(m, e, n)
}

fn rsa_dec(n: i128, d: i128, m: i128) -> i128 {
    modpow(m, d, n)
}

fn order(g: i128, p: i128) -> i128 {
    let mut i = 1;
    let mut n = g;
    while n != 1 {
        n = (n * g) % p;
        i += 1;
    }
    i
}

fn dsa_is_pub_key(p: i128, q: i128, b: i128) -> bool {
    for g in 2..p {
        if order(g, p) == q {
            for d in 2..q {
                if modpow(g, d, p) == b {
                    return true;
                }
            }
        }
    }
    false
}

fn is_ec(a: i128, b: i128, p: i128) -> bool {
    let d = (4 * a.pow(3) + 27 * b.pow(2)) % p;
    d != 0
}

fn ec_points(a: i128, b: i128, p: i128) -> Vec<(i128, i128)> {
    let mut points = vec![];
    for x in 0..p {
        let y2 = (x.pow(3) + a * x + b) % p;
        for y in 0..p {
            if y.pow(2) % p == y2 {
                points.push((x, y));
            }
        }
    }
    points.push((0, 0)); // infinity
    points
}

fn is_ec_point(a: i128, b: i128, p: i128, x: i128, y: i128) -> bool {
    let y2 = (x.pow(3) + a * x + b) % p;
    y.pow(2) % p == y2
}

fn ec_add(a: i128, _b: i128, p: i128, x1: i128, y1: i128, x2: i128, y2: i128) -> (i128, i128) {
    if (x1, y1) == (0, 0) {
        return (x2, y2);
    }
    if (x2, y2) == (0, 0) {
        return (x1, y1);
    }
    let s = if x1 == x2 && y1 == y2 {
        // point doubling
        if y1 == 0 {
            return (0, 0);
        }
        moddiv(3 * x1.pow(2) + a, 2 * y1, p)
    } else {
        // point addition
        if x1 == x2 {
            return (0, 0);
        }
        moddiv(y2 - y1, x2 - x1, p)
    };
    let x3 = modulus(s.pow(2) - x1 - x2, p);
    let y3 = modulus(s * (x1 - x3) - y1, p);
    (x3, y3)
}

fn ec_double(a: i128, _b: i128, p: i128, x: i128, y: i128) -> (i128, i128) {
    ec_add(a, 0, p, x, y, x, y)
}

fn ec_order(a: i128, b: i128, p: i128, x: i128, y: i128) -> i128 {
    let mut points = Vec::new();
    let mut prev = (x, y);
    points.push(prev);
    loop {
        prev = ec_add(a, b, p, prev.0, prev.1, x, y);
        if points.contains(&prev) {
            break;
        }
        points.push(prev);
    }
    points.len() as i128
}

fn dsa_sig(p: i128, q: i128, g: i128, d: i128, k: i128, m: i128) -> (i128, i128) {
    let r = modpow(g, k, p) % q;
    let s = moddiv(m + r * d, k, q);
    (r, s)
}

fn dsa_ver(p: i128, q: i128, g: i128, b: i128, r: i128, s: i128, m: i128) -> bool {
    let w = modulus(modinv(s, q), q);
    let u1 = w * m % q;
    let u2 = w * r % q;
    let v = modpow(g, u1, p) * modpow(b, u2, p) % p % q;
    v == r
}

fn elgamal_sig(p: i128, g: i128, d: i128, k: i128, m: i128) -> (i128, i128) {
    let r = modpow(g, k, p);
    let s = moddiv(m - d * r, k, p - 1);
    (r, s)
}

fn elgamal_ver(p: i128, g: i128, b: i128, r: i128, s: i128, m: i128) -> bool {
    let t = modpow(b, r, p) * modpow(r, s, p) % p;
    t == modpow(g, m, p)
}

// Alice is encrypting a message m to Bob using Elgamal encryption
// exp is the random exponent chosen by Alice
// returns the ciphertext (ka, c)
fn elgamal_enc(p: i128, g: i128, kb: i128, exp: i128, m: i128) -> (i128, i128) {
    let ka = modpow(g, exp, p);
    let kab = modpow(kb, exp, p);
    (ka, (m * kab) % p)
}

// Bob is decrypting a ciphertext c from Alice using Elgamal decryption
// exp is the random exponent chosen by Bob used to generate kb
fn elgamal_dec(p: i128, ka: i128, exp: i128, c: i128) -> i128 {
    let kab = modpow(ka, exp, p);
    c * modinv(kab, p) % p
}

fn des_enc(key: u64, plaintext: u64) -> u64 {
    let des = Des::new(Key::<Des>::from_slice(&key.to_be_bytes()));
    let mut block = Block::<Des>::from(plaintext.to_be_bytes());
    des.encrypt_block(&mut block);
    u64::from_be_bytes(block.into())
}

fn des_dec(key: u64, ciphertext: u64) -> u64 {
    let des = Des::new(Key::<Des>::from_slice(&key.to_be_bytes()));
    let mut block = Block::<Des>::from(ciphertext.to_be_bytes());
    des.decrypt_block(&mut block);
    u64::from_be_bytes(block.into())
}

fn aes_enc(key: u128, plaintext: u128) -> u128 {
    let cipher = aes::Aes128::new(Key::<Aes128>::from_slice(&key.to_be_bytes()));
    let mut block = Block::<Aes128>::from(plaintext.to_be_bytes());
    cipher.encrypt_block(&mut block);
    u128::from_be_bytes(block.into())
}

fn aes_dec(key: u128, ciphertext: u128) -> u128 {
    let cipher = aes::Aes128::new(Key::<Aes128>::from_slice(&key.to_be_bytes()));
    let mut block = Block::<Aes128>::from(ciphertext.to_be_bytes());
    cipher.decrypt_block(&mut block);
    u128::from_be_bytes(block.into())
}

// this could be a more efficient algorithm, but our exam numbers will be small so this will be
// fine
fn dlp(g: i128, b: i128, p: i128) -> i128 {
    let mut x = 1;
    let mut a = g;
    while a != b && x < p {
        a = (a * g) % p;
        x += 1;
    }
    if x == p {
        panic!("No solution found");
    }
    x
}

#[derive(clap::Parser)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(clap::Subcommand)]
enum Command {
    #[command(name = "eval")]
    /// Evaluate an expression modulo m
    /// The expression can contain numbers, operators (+, -, *, /, ^), and parentheses
    /// For example, exam eval 7 "5 + 3 * (2 - 1) ^ 2" and exam eval 3 2 ^ 3 ^ 2 will evaluate to 1
    /// If using parentheses, you may need to quote the expression to prevent the shell from
    /// getting cranky
    Eval {
        /// The modulus
        m: i128,
        /// The expression to evaluate
        #[arg(trailing_var_arg = true)]
        expr: Vec<String>,
    },
    #[command(name = "modinv")]
    /// Calculate the modular inverse a^-1 mod m
    ModInv {
        /// The number to calculate the modular inverse of
        a: i128,
        /// The modulus
        m: i128,
    },
    #[command(name = "moddiv")]
    /// Calculate the modular division a/b mod m
    ModDiv {
        /// The dividend
        a: i128,
        /// The divisor
        b: i128,
        /// The modulus
        m: i128,
    },
    #[command(name = "modpow")]
    /// Calculate the modular exponentiation base^exp mod modulus
    ModPow {
        /// The base
        base: i128,
        /// The exponent
        exp: i128,
        /// The modulus
        modulus: i128,
    },
    #[command(name = "gen")]
    /// Check if g is a generator of the group Z_p
    Generator {
        /// The generator
        g: i128,
        /// The prime modulus
        p: i128,
    },
    #[command(name = "pf")]
    /// Calculate the prime factors of p
    PrimeFactors { p: i128 },
    #[command(name = "gcd")]
    /// Calculate the greatest common divisor of a and b
    Gcd { a: i128, b: i128 },
    #[command(name = "ee")]
    /// Calculate the extended Euclidean algorithm for a and b
    ExtendedEuclid { a: i128, b: i128 },
    #[command(name = "rsa-sig")]
    /// Calculate the RSA signature of m
    RsaSign {
        /// The modulus
        n: i128,
        /// The private exponent
        d: i128,
        /// The message
        m: i128,
    },
    #[command(name = "rsa-ver")]
    /// Verify the RSA signature of m
    RsaVer {
        /// The modulus
        #[arg(short)]
        n: i128,
        /// The public exponent
        #[arg(short)]
        e: i128,
        /// The message
        #[arg(short)]
        m: i128,
        /// The signature
        #[arg(short)]
        s: i128,
    },
    /// Calculate the RSA encryption of m
    #[command(name = "rsa-enc")]
    RsaEnc {
        /// The modulus
        #[arg(short)]
        n: i128,
        /// The public exponent
        #[arg(short)]
        e: i128,
        /// The message
        #[arg(short)]
        m: i128,
    },
    /// Calculate the RSA decryption of m
    #[command(name = "rsa-dec")]
    RsaDec {
        /// The modulus
        #[arg(short)]
        n: i128,
        /// The private exponent
        #[arg(short)]
        d: i128,
        /// The message
        #[arg(short)]
        m: i128,
    },
    #[command(name = "order")]
    /// Calculate the order of g in Z_p
    Order { g: i128, p: i128 },
    #[command(name = "is-dsa-pub")]
    /// Check if b is a public key for DSA
    DsaIsPubKey { p: i128, q: i128, b: i128 },
    #[command(name = "is-ec")]
    /// Check if the curve is an elliptic curve
    IsEc { a: i128, b: i128, p: i128 },
    #[command(name = "ec-points")]
    /// Calculate the points on the elliptic curve
    EcPoints { a: i128, b: i128, p: i128 },
    #[command(name = "is-ec-point")]
    /// Check if the point is on the elliptic curve
    IsEcPoint {
        /// The curve parameter a
        #[arg(short)]
        a: i128,
        /// The curve parameter b
        #[arg(short)]
        b: i128,
        /// The curve parameter p
        #[arg(short)]
        p: i128,
        /// The x-coordinate of the point
        #[arg(short)]
        x: i128,
        /// The y-coordinate of the point
        #[arg(short)]
        y: i128,
    },
    #[command(name = "ec-add")]
    /// Add two points on the elliptic curve
    EcAdd {
        /// The curve parameter a
        #[arg(short)]
        a: i128,
        /// The curve parameter b
        #[arg(short)]
        b: i128,
        /// The curve parameter p
        #[arg(short)]
        p: i128,
        /// The x-coordinate of the first point
        #[arg(long)]
        x1: i128,
        /// The y-coordinate of the first point
        #[arg(long)]
        y1: i128,
        /// The x-coordinate of the second point
        #[arg(long)]
        x2: i128,
        /// The y-coordinate of the second point
        #[arg(long)]
        y2: i128,
    },
    #[command(name = "ec-double")]
    /// Double a point on the elliptic curve
    EcDouble {
        /// The curve parameter a
        #[arg(short)]
        a: i128,
        /// The curve parameter b
        #[arg(short)]
        b: i128,
        /// The curve parameter p
        #[arg(short)]
        p: i128,
        /// The x-coordinate of the point
        #[arg(short)]
        x: i128,
        /// The y-coordinate of the point
        #[arg(short)]
        y: i128,
    },
    #[command(name = "ec-order")]
    /// Calculate the order of the point on the elliptic curve
    EcOrder {
        /// The curve parameter a
        #[arg(short)]
        a: i128,
        /// The curve parameter b
        #[arg(short)]
        b: i128,
        /// The curve parameter p
        #[arg(short)]
        p: i128,
        /// The x-coordinate of the point
        #[arg(short)]
        x: i128,
        /// The y-coordinate of the point
        #[arg(short)]
        y: i128,
    },
    #[command(name = "dsa-sig")]
    /// Calculate the DSA signature of m
    DsaSign {
        /// The prime p
        #[arg(short)]
        p: i128,
        /// The prime q
        #[arg(short)]
        q: i128,
        /// The generator
        #[arg(short)]
        g: i128,
        /// The private key
        #[arg(short)]
        d: i128,
        /// The random number
        #[arg(short)]
        k: i128,
        /// The message
        #[arg(short)]
        m: i128,
    },
    #[command(name = "dsa-ver")]
    /// Verify the DSA signature of m
    DsaVer {
        /// The prime p
        #[arg(short)]
        p: i128,
        /// The prime q
        #[arg(short)]
        q: i128,
        /// The generator
        #[arg(short)]
        g: i128,
        /// The public key
        #[arg(short)]
        b: i128,
        /// The signature r
        #[arg(short)]
        r: i128,
        /// The signature s
        #[arg(short)]
        s: i128,
        /// The message
        #[arg(short)]
        m: i128,
    },
    #[command(name = "elg-sig")]
    /// Calculate the Elgamal signature of m
    ElgamalSig {
        /// The prime p
        #[arg(short)]
        p: i128,
        /// The generator
        #[arg(short)]
        g: i128,
        /// The private key
        #[arg(short)]
        d: i128,
        /// The random number
        #[arg(short)]
        k: i128,
        /// The message
        #[arg(short)]
        m: i128,
    },
    #[command(name = "elg-ver")]
    /// Verify the Elgamal signature of m
    ElgamalVer {
        /// The prime p
        #[arg(short)]
        p: i128,
        /// The generator
        #[arg(short)]
        g: i128,
        /// The public key
        #[arg(short)]
        b: i128,
        /// The signature r
        #[arg(short)]
        r: i128,
        /// The signature s
        #[arg(short)]
        s: i128,
        /// The message
        #[arg(short)]
        m: i128,
    },
    #[command(name = "elg-enc")]
    /// Calculate the Elgamal encryption of m
    /// Alice is encrypting a message m to Bob
    ElgamalEnc {
        /// The prime p
        #[arg(short)]
        p: i128,
        /// The generator
        #[arg(short)]
        g: i128,
        /// This is g^d mod p, where d is the exponent chosen by Bob
        /// This is the public key of Bob
        #[arg(short)]
        kb: i128,
        /// The random exponent chosen by Alice
        #[arg(short)]
        exp: i128,
        /// The message
        #[arg(short)]
        m: i128,
    },
    #[command(name = "elg-dec")]
    /// Calculate the Elgamal decryption of c
    /// Bob is decrypting a ciphertext c from Alice
    ElgamalDec {
        /// The prime p
        #[arg(short)]
        p: i128,
        /// This is the ka chosen by Alice (sent in the ciphertext)
        #[arg(short)]
        ka: i128,
        /// The random exponent chosen by Bob used to generate kb
        #[arg(short)]
        exp: i128,
        /// The ciphertext
        #[arg(short)]
        c: i128,
    },
    #[command(name = "sha256")]
    /// Calculate the SHA-256 hash of a message
    /// One of msg or num must be provided
    Sha256 {
        /// The message to hash
        #[arg(short)]
        msg: Option<String>,
        /// The number to hash, big-endian 128-bit integer
        #[arg(short)]
        num: Option<i128>,
    },
    #[command(name = "lfsr-gen")]
    /// Generate bits using a linear feedback shift register
    LfsrGen {
        /// The coefficients of the LFSR, i.e. 1011
        coefs: String,
        /// The initial state of the LFSR, i.e. 1010
        init: String,
        /// The number of bits to generate
        num: usize,
    },
    #[command(name = "lfsr-enc")]
    /// Encrypt a message using a linear feedback shift register
    LfsrEnc {
        /// The coefficients of the LFSR, i.e. 1011
        coefs: String,
        /// The initial state of the LFSR, i.e. 1010
        init: String,
        /// The message to encrypt, i.e. 10101010
        msg: String,
    },
    #[command(name = "lfsr-dec")]
    /// Decrypt a message using a linear feedback shift register
    LfsrDec {
        /// The coefficients of the LFSR, i.e. 1011
        coefs: String,
        /// The initial state of the LFSR, i.e. 1010
        init: String,
        /// The message to decrypt, i.e. 10101010
        msg: String,
    },
    #[command(name = "lfsr-solve")]
    /// Solve for the coefficients of an LFSR given the output
    LfsrSolve {
        /// The number of coefficients
        coefs: usize,
        /// The output of the LFSR, must be 2 * coefs in length, i.e. 10101010
        output: String,
    },
    #[command(name = "des-enc")]
    /// Encrypt a message using the Data Encryption Standard
    /// The key and plaintext must be 64-bit unsigned integers
    DesEnc {
        /// The key
        key: u64,
        /// The plaintext
        plaintext: u64,
    },
    #[command(name = "des-dec")]
    /// Decrypt a message using the Data Encryption Standard
    /// The key and ciphertext must be 64-bit unsigned integers
    DesDec {
        /// The key
        key: u64,
        /// The ciphertext
        ciphertext: u64,
    },
    #[command(name = "aes-enc")]
    /// Encrypt a message using the Advanced Encryption Standard
    /// The key and plaintext must be 128-bit unsigned integers
    AesEnc {
        /// The key
        key: u128,
        /// The plaintext
        plaintext: u128,
    },
    #[command(name = "aes-dec")]
    /// Decrypt a message using the Advanced Encryption Standard
    /// The key and ciphertext must be 128-bit unsigned integers
    AesDec {
        /// The key
        key: u128,
        /// The ciphertext
        ciphertext: u128,
    },
    #[command(name = "dlp")]
    /// Solve the discrete logarithm problem g^x = b mod p
    Dlp {
        /// The generator
        g: i128,
        /// The base
        b: i128,
        /// The modulus
        p: i128,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Command::Eval { m, expr } => {
            let expr = Expr::parse_str(&expr.join(" "));
            match expr {
                Ok(expr) => {
                    println!("Evaluated expression (mod {}):", m);
                    println!("{}", expr);
                    println!("Result: {}", expr.eval(m));
                }
                Err(e) => {
                    eprintln!("{}", e);
                }
            }
        }
        Command::ModInv { a, m } => {
            println!("{}", modinv(a, m));
        }
        Command::ModDiv { a, b, m } => {
            println!("{}", moddiv(a, b, m));
        }
        Command::ModPow { base, exp, modulus } => {
            println!("{}", modpow(base, exp, modulus));
        }
        Command::Generator { g, p } => {
            println!("{}", is_generator(g, p));
        }
        Command::PrimeFactors { p } => {
            let factors = prime_factors(p);
            for (k, v) in factors {
                println!("{}^{}", k, v);
            }
        }
        Command::Gcd { a, b } => {
            println!("{}", gcd(a, b));
        }
        Command::ExtendedEuclid { a, b } => {
            let (gcd, s, t) = extended_euclid(a, b);
            println!("gcd = {}, s = {}, t = {}", gcd, s, t);
        }
        Command::RsaSign { n, d, m } => {
            println!("{}", rsa_sig(n, d, m));
        }
        Command::RsaVer { n, e, m, s } => {
            println!("{}", rsa_ver(n, e, m, s));
        }
        Command::RsaEnc { n, e, m } => {
            println!("{}", rsa_enc(n, e, m));
        }
        Command::RsaDec { n, d, m } => {
            println!("{}", rsa_dec(n, d, m));
        }
        Command::Order { g, p } => {
            println!("{}", order(g, p));
        }
        Command::DsaIsPubKey { p, q, b } => {
            println!("{}", dsa_is_pub_key(p, q, b));
        }
        Command::IsEc { a, b, p } => {
            println!("{}", is_ec(a, b, p));
        }
        Command::EcPoints { a, b, p } => {
            let points = ec_points(a, b, p);
            for (x, y) in &points {
                println!("({}, {})", x, y);
            }
            println!("Total (with infinity): {}", points.len());
        }
        Command::IsEcPoint { a, b, p, x, y } => {
            println!("{}", is_ec_point(a, b, p, x, y));
        }
        Command::EcAdd {
            a,
            b,
            p,
            x1,
            y1,
            x2,
            y2,
        } => {
            let (x3, y3) = ec_add(a, b, p, x1, y1, x2, y2);
            println!("({}, {})", x3, y3);
        }
        Command::EcDouble { a, b, p, x, y } => {
            let (x3, y3) = ec_double(a, b, p, x, y);
            println!("({}, {})", x3, y3);
        }
        Command::EcOrder { a, b, p, x, y } => {
            println!("{}", ec_order(a, b, p, x, y));
        }
        Command::DsaSign { p, q, g, d, k, m } => {
            let (r, s) = dsa_sig(p, q, g, d, k, m);
            println!("({}, {})", r, s);
        }
        Command::DsaVer {
            p,
            q,
            g,
            b,
            r,
            s,
            m,
        } => {
            println!("{}", dsa_ver(p, q, g, b, r, s, m));
        }
        Command::ElgamalSig { p, g, d, k, m } => {
            let (r, s) = elgamal_sig(p, g, d, k, m);
            println!("({}, {})", r, s);
        }
        Command::ElgamalVer { p, g, b, r, s, m } => {
            println!("{}", elgamal_ver(p, g, b, r, s, m));
        }
        Command::ElgamalEnc { p, g, kb, exp, m } => {
            let (ka, c) = elgamal_enc(p, g, kb, exp, m);
            println!("(ka = {}, c = {})", ka, c);
        }
        Command::ElgamalDec { p, ka, exp, c } => {
            println!("{}", elgamal_dec(p, ka, exp, c));
        }
        Command::Sha256 { num, msg } => {
            let mut hasher = <sha2::Sha256 as Digest>::new();
            if num.is_none() && msg.is_none() {
                panic!("One of num or msg must be provided");
            }
            if num.is_some() && msg.is_some() {
                panic!("Only one of num or msg can be provided");
            }
            if let Some(num) = num {
                hasher.update(num.to_be_bytes());
            }
            if let Some(msg) = msg {
                hasher.update(msg);
            }
            let result = hasher.finalize();
            println!("{:x}", result);
        }
        Command::LfsrGen { coefs, init, num } => {
            let mut lfsr = Lfsr::new(read_binary_string(&coefs), read_binary_string(&init));
            let bits = (0..num).map(|_| lfsr.next()).collect::<Vec<_>>();
            let mut bin = to_binary_string(&bits);
            bin.insert(8, ' ');
            println!("{}", bin);
        }
        Command::LfsrEnc { coefs, init, msg } => {
            let mut lfsr = Lfsr::new(read_binary_string(&coefs), read_binary_string(&init));
            let bits = read_binary_string(&msg);
            let enc = lfsr.encrypt(&bits);
            let mut bin = to_binary_string(&enc);
            bin.insert(8, ' ');
            println!("{}", bin);
        }
        Command::LfsrDec { coefs, init, msg } => {
            let mut lfsr = Lfsr::new(read_binary_string(&coefs), read_binary_string(&init));
            let bits = read_binary_string(&msg);
            let dec = lfsr.decrypt(&bits);
            let mut bin = to_binary_string(&dec);
            bin.insert(8, ' ');
            println!("{}", bin);
        }
        Command::LfsrSolve { coefs, output } => {
            let output = read_binary_string(&output);
            let coefs = solve_for_coefs(coefs, &output);
            let mut bin = to_binary_string(&coefs);
            bin.insert(8, ' ');
            println!("{}", bin);
        }
        Command::DesEnc { key, plaintext } => {
            println!("{}", des_enc(key, plaintext));
        }
        Command::DesDec { key, ciphertext } => {
            println!("{}", des_dec(key, ciphertext));
        }
        Command::AesEnc { key, plaintext } => {
            println!("{}", aes_enc(key, plaintext));
        }
        Command::AesDec { key, ciphertext } => {
            println!("{}", aes_dec(key, ciphertext));
        }
        Command::Dlp { g, b, p } => {
            println!("{}", dlp(g, b, p));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_modinv() {
        assert_eq!(modinv(3, 11), 4);
        assert_eq!(modinv(7, 11), 8);
        assert_eq!(modinv(9, 11), 5);
        assert_eq!(modinv(5, 9), 2);
    }

    #[test]
    fn test_moddiv() {
        assert_eq!(moddiv(3, 5, 9), 6);
    }

    #[test]
    fn test_modpow() {
        assert_eq!(modpow(2, 3, 5), 3);
        assert_eq!(modpow(2, 10, 7), 2);
        assert_eq!(modpow(2, 10, 11), 1);
        assert_eq!(modpow(2, 10, 13), 10);
    }

    #[test]
    fn test_is_generator() {
        assert!(!is_generator(1, 11));
        assert!(is_generator(2, 11));
        assert!(!is_generator(3, 11));
        assert!(!is_generator(4, 11));
        assert!(!is_generator(5, 11));
        assert!(is_generator(6, 11));
        assert!(is_generator(7, 11));
        assert!(is_generator(8, 11));
        assert!(!is_generator(9, 11));
        assert!(!is_generator(10, 11));
    }

    #[test]
    fn test_prime_factors() {
        let factors = prime_factors(240);
        assert_eq!(factors.get(&2), Some(&4));
        assert_eq!(factors.get(&3), Some(&1));
        assert_eq!(factors.get(&5), Some(&1));

        let factors = prime_factors(46);
        assert_eq!(factors.get(&2), Some(&1));
        assert_eq!(factors.get(&23), Some(&1));
    }

    #[test]
    fn test_gcd() {
        assert_eq!(gcd(10, 15), 5);
        assert_eq!(gcd(130, 52), 26);
        assert_eq!(gcd(52, 130), 26);
        assert_eq!(gcd(27, 21), 3);
        assert_eq!(gcd(21, 27), 3);
    }

    #[test]
    fn test_extended_euclid() {
        let (gcd, s, t) = extended_euclid(10, 15);
        assert_eq!(gcd, 5);
        assert_eq!(s * 10 + t * 15, 5);
        assert_eq!(s, -1);
        assert_eq!(t, 1);

        let (gcd, s, t) = extended_euclid(130, 52);
        assert_eq!(gcd, 26);
        assert_eq!(s * 130 + t * 52, 26);
        assert_eq!(s, 1);
        assert_eq!(t, -2);
    }

    #[test]
    fn test_rsa_sig() {
        assert_eq!(rsa_sig(85, 57, 6), 11);
    }

    #[test]
    fn test_rsa_ver() {
        assert!(rsa_ver(85, 9, 6, 11));
        assert!(!rsa_ver(85, 9, 6, 12));
    }

    #[test]
    fn test_rsa_enc() {
        assert_eq!(rsa_enc(85, 9, 23), 28);
    }

    #[test]
    fn test_rsa_dec() {
        assert_eq!(rsa_dec(85, 57, 28), 23);
    }

    #[test]
    fn test_rsa_enc_roundtrip() {
        let n = 85;
        let e = 9;
        let m = 23;
        let c = rsa_enc(n, e, m);
        assert_eq!(rsa_dec(n, 57, c), m);
    }

    #[test]
    fn test_rsa_sig_roundtrip() {
        let n = 85;
        let d = 57;
        let m = 6;
        let s = rsa_sig(n, d, m);
        assert!(rsa_ver(n, 9, m, s));
    }

    #[test]
    fn test_order() {
        assert_eq!(order(1, 11), 1);
        assert_eq!(order(2, 11), 10);
        assert_eq!(order(3, 11), 5);
        assert_eq!(order(4, 11), 5);
        assert_eq!(order(5, 11), 5);
        assert_eq!(order(6, 11), 10);
        assert_eq!(order(7, 11), 10);
        assert_eq!(order(8, 11), 10);
        assert_eq!(order(9, 11), 5);
        assert_eq!(order(10, 11), 2);
    }

    #[test]
    fn test_dsa_is_pub_key() {
        assert!(dsa_is_pub_key(53, 13, 24));
        assert!(!dsa_is_pub_key(53, 13, 25));
    }

    #[test]
    fn test_is_ec() {
        assert!(is_ec(2, 2, 17));
        assert!(!is_ec(0, 0, 17));
        assert!(is_ec(1, 1, 17));
        assert!(is_ec(2, 3, 17));
    }

    #[test]
    fn test_ec_points() {
        let points = ec_points(2, 2, 17);
        assert_eq!(points.len(), 19);
        let correct_points = [
            (6, 3),
            (10, 6),
            (3, 1),
            (9, 16),
            (16, 13),
            (0, 6),
            (13, 7),
            (7, 6),
            (7, 11),
            (13, 10),
            (0, 11),
            (16, 4),
            (9, 1),
            (3, 16),
            (10, 11),
            (6, 14),
            (5, 16),
            (0, 0),
        ];
        for p in correct_points {
            assert!(points.contains(&p));
        }
    }

    #[test]
    fn test_is_ec_point() {
        assert!(is_ec_point(2, 2, 17, 6, 3));
        assert!(!is_ec_point(2, 2, 17, 6, 4));
    }

    #[test]
    fn test_ec_add() {
        assert_eq!(ec_add(2, 2, 17, 5, 1, 6, 3), (10, 6));
    }

    #[test]
    fn test_ec_double() {
        assert_eq!(ec_double(2, 2, 17, 5, 1), (6, 3));
    }

    #[test]
    fn test_ec_order() {
        assert_eq!(ec_order(2, 2, 17, 5, 1), 19);
    }

    #[test]
    fn test_dsa_sig() {
        assert_eq!(dsa_sig(53, 13, 10, 8, 9, 6), (2, 1));
    }

    #[test]
    fn test_dsa_ver() {
        assert!(dsa_ver(53, 13, 10, 24, 2, 1, 6));
        assert!(!dsa_ver(53, 13, 10, 24, 3, 1, 6));
    }

    #[test]
    fn test_dsa_roundtrip() {
        let p = 53;
        let q = 13;
        let g = 10;
        let d = 8;
        let k = 9;
        let m = 6;
        let (r, s) = dsa_sig(p, q, g, d, k, m);
        assert!(dsa_ver(p, q, g, modpow(g, d, p), r, s, m));
    }

    #[test]
    fn test_elgamal_sig() {
        assert_eq!(elgamal_sig(53, 27, 25, 19, 41), (31, 38));
    }

    #[test]
    fn test_elgamal_ver() {
        assert!(elgamal_ver(53, 27, 51, 31, 38, 41));
        assert!(!elgamal_ver(53, 27, 51, 31, 39, 41));
    }

    #[test]
    fn test_elgamal_sig_roundtrip() {
        let p = 53;
        let g = 27;
        let d = 25;
        let k = 19;
        let m = 41;
        let (r, s) = elgamal_sig(p, g, d, k, m);
        assert!(elgamal_ver(p, g, modpow(g, d, p), r, s, m));
    }

    #[test]
    fn test_elgamal_enc() {
        assert_eq!(elgamal_enc(53, 27, 46, 32, 21), (24, 34));
        assert_eq!(elgamal_enc(467, 2, 444, 213, 33), (29, 296));
    }

    #[test]
    fn test_elgamal_dec() {
        assert_eq!(elgamal_dec(53, 24, 12, 34), 21);
        assert_eq!(elgamal_dec(467, 29, 105, 296), 33);
    }

    #[test]
    fn test_elgamal_enc_dec() {
        let p = 53;
        let g = 27;
        let d = 46;
        let exp = 32;
        let m = 21;
        let (ka, c) = elgamal_enc(p, g, modpow(g, d, p), exp, m);
        assert_eq!(elgamal_dec(p, ka, d, c), m);

        let p = 467;
        let g = 2;
        let d = 105;
        let exp = 213;
        let m = 33;
        let (ka, c) = elgamal_enc(p, g, modpow(g, d, p), exp, m);
        assert_eq!(elgamal_dec(p, ka, d, c), m);
    }

    #[test]
    fn test_des_roundtrip() {
        assert_eq!(
            des_dec(
                0x133457799BBCDFF1,
                des_enc(0x133457799BBCDFF1, 0x0123456789ABCDEF)
            ),
            0x0123456789ABCDEF
        );
    }

    #[test]
    fn test_aes_roundtrip() {
        assert_eq!(
            aes_dec(
                0x2b7e151628aed2a6abf7158809cf4f3c,
                aes_enc(
                    0x2b7e151628aed2a6abf7158809cf4f3c,
                    0x3243f6a8885a308d313198a2e0370734
                )
            ),
            0x3243f6a8885a308d313198a2e0370734
        );
    }

    #[test]
    fn test_dlp() {
        assert_eq!(dlp(5, 41, 47), 15);
        assert_eq!(dlp(2, 36, 47), 17);
    }
}
