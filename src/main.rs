use std::collections::{HashMap, HashSet};

use clap::Parser;

fn modulus(a: i128, m: i128) -> i128 {
    let r = a % m;
    if r < 0 {
        r + m
    } else {
        r
    }
}

fn modinv(a: i128, m: i128) -> i128 {
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
        if (x1, x2) == (x2, -y2) {
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
    let s = moddiv(k, modinv(m - r * d, q), q);
    (r, s)
}

fn dsa_ver(p: i128, q: i128, g: i128, b: i128, r: i128, s: i128, m: i128) -> bool {
    let w = modinv(s, q);
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

#[derive(clap::Parser)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(clap::Subcommand)]
enum Command {
    #[command(name = "modinv")]
    ModInv { a: i128, m: i128 },
    #[command(name = "moddiv")]
    ModDiv { a: i128, b: i128, m: i128 },
    #[command(name = "modpow")]
    ModPow {
        base: i128,
        exp: i128,
        modulus: i128,
    },
    #[command(name = "gen")]
    Generator { g: i128, p: i128 },
    #[command(name = "pf")]
    PrimeFactors { p: i128 },
    #[command(name = "gcd")]
    Gcd { a: i128, b: i128 },
    #[command(name = "ee")]
    ExtendedEuclid { a: i128, b: i128 },
    #[command(name = "rsa-sig")]
    RsaSign { n: i128, d: i128, m: i128 },
    #[command(name = "rsa-ver")]
    RsaVer { n: i128, e: i128, m: i128, s: i128 },
    #[command(name = "order")]
    Order { g: i128, p: i128 },
    #[command(name = "is-dsa-pub")]
    DsaIsPubKey { p: i128, q: i128, b: i128 },
    #[command(name = "is-ec")]
    IsEc { a: i128, b: i128, p: i128 },
    #[command(name = "ec-points")]
    EcPoints { a: i128, b: i128, p: i128 },
    #[command(name = "is-ec-point")]
    IsEcPoint {
        a: i128,
        b: i128,
        p: i128,
        x: i128,
        y: i128,
    },
    #[command(name = "ec-add")]
    EcAdd {
        a: i128,
        b: i128,
        p: i128,
        x1: i128,
        y1: i128,
        x2: i128,
        y2: i128,
    },
    #[command(name = "ec-double")]
    EcDouble {
        a: i128,
        b: i128,
        p: i128,
        x: i128,
        y: i128,
    },
    #[command(name = "ec-order")]
    EcOrder {
        a: i128,
        b: i128,
        p: i128,
        x: i128,
        y: i128,
    },
    #[command(name = "dsa-sig")]
    DsaSign {
        p: i128,
        q: i128,
        g: i128,
        d: i128,
        k: i128,
        m: i128,
    },
    #[command(name = "dsa-ver")]
    DsaVer {
        p: i128,
        q: i128,
        g: i128,
        b: i128,
        r: i128,
        s: i128,
        m: i128,
    },
    #[command(name = "elg-sig")]
    ElgamalSig {
        p: i128,
        g: i128,
        d: i128,
        k: i128,
        m: i128,
    },
    #[command(name = "elg-ver")]
    ElgamalVer {
        p: i128,
        g: i128,
        b: i128,
        r: i128,
        s: i128,
        m: i128,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
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
    fn test_elgamal_sig() {
        assert_eq!(elgamal_sig(53, 27, 25, 19, 41), (31, 38));
    }

    #[test]
    fn test_elgamal_ver() {
        assert!(elgamal_ver(53, 27, 51, 31, 38, 41));
        assert!(!elgamal_ver(53, 27, 51, 31, 39, 41));
    }
}
