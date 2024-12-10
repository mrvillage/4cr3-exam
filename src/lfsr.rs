use faer::solvers::SpSolver;
use std::fmt::Write;

#[derive(Debug, Clone)]
pub struct Lfsr {
    coefs: Vec<u8>,
    init: Vec<u8>,
    state: Vec<u8>,
    next: usize,
}

impl Lfsr {
    pub fn new(coefs: Vec<u8>, state: Vec<u8>) -> Self {
        Self {
            coefs,
            init: state.clone(),
            state,
            next: 0,
        }
    }

    pub fn next(&mut self) -> u8 {
        if self.next < self.coefs.len() {
            self.next += 1;
            return self.state[self.next - 1];
        }
        let mut next = 0;
        for (v, c) in self.state.iter().zip(self.coefs.iter()) {
            next += v * c;
        }
        next %= 2;
        self.state.rotate_left(1);
        self.state[self.coefs.len() - 1] = next;
        next
    }

    pub fn encrypt(&mut self, plaintext: &[u8]) -> Vec<u8> {
        let mut ciphertext = Vec::with_capacity(plaintext.len());
        for p in plaintext.iter() {
            ciphertext.push(p ^ self.next());
        }
        ciphertext
    }

    pub fn decrypt(&mut self, ciphertext: &[u8]) -> Vec<u8> {
        self.encrypt(ciphertext)
    }

    pub fn reset(&mut self) {
        self.state = self.init.clone();
        self.next = 0;
    }
}

// output must be length 2 * coefs
pub fn solve_for_coefs(coefs: usize, output: &[u8]) -> Vec<u8> {
    let output = output.iter().map(|c| *c as f64).collect::<Vec<_>>();
    let mut rows = Vec::new();
    for i in 0..coefs {
        rows.push(output[i..i + coefs].to_vec());
    }
    let rows = rows.concat();
    let mat = faer::mat::from_row_major_slice(&rows, coefs, coefs);
    let nexts = faer::col::from_slice(&output[coefs..]);
    let coefs = mat.full_piv_lu().solve(&nexts);
    coefs.iter().map(|c| c.abs() as u8).collect()
}

pub fn read_binary_string(s: &str) -> Vec<u8> {
    s.chars().map(|c| c.to_digit(2).unwrap() as u8).collect()
}

pub fn to_binary_string(v: &[u8]) -> String {
    v.iter().fold(String::new(), |mut acc, c| {
        write!(acc, "{}", c).unwrap();
        acc
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::identity_op)]
    fn test_lfsr_encrypt() {
        let coefs = vec![1, 0, 1, 1];
        let state = vec![1, 0, 1, 0];
        let mut lfsr = Lfsr::new(coefs, state);
        let plaintext = [0b1, 0b0, 0b1, 0b0, 0b0, 0b0, 0b1, 0b1];
        let ciphertext = lfsr.encrypt(&plaintext);
        assert_eq!(
            ciphertext,
            vec![
                plaintext[0] ^ 1,
                plaintext[1] ^ 0,
                plaintext[2] ^ 1,
                plaintext[3] ^ 0,
                plaintext[4] ^ 0,
                plaintext[5] ^ 0,
                plaintext[6] ^ 1,
                plaintext[7] ^ 1
            ]
        );
    }

    #[test]
    #[allow(clippy::identity_op)]
    fn test_lfsr_decrypt() {
        let coefs = vec![1, 0, 1, 1];
        let state = vec![1, 0, 1, 0];
        let mut lfsr = Lfsr::new(coefs, state);
        let ciphertext = [0b0, 0b1, 0b0, 0b1, 0b0, 0b0, 0b1, 0b1];
        let plaintext = lfsr.decrypt(&ciphertext);
        assert_eq!(
            plaintext,
            vec![
                ciphertext[0] ^ 1,
                ciphertext[1] ^ 0,
                ciphertext[2] ^ 1,
                ciphertext[3] ^ 0,
                ciphertext[4] ^ 0,
                ciphertext[5] ^ 0,
                ciphertext[6] ^ 1,
                ciphertext[7] ^ 1
            ]
        );
    }

    #[test]
    fn test_lfsr_roundtrip() {
        let coefs = vec![1, 0, 1, 1];
        let state = vec![1, 0, 1, 0];
        let mut lfsr = Lfsr::new(coefs, state);
        let plaintext = [0b1, 0b0, 0b1, 0b0, 0b0, 0b0, 0b1, 0b1];
        let ciphertext = lfsr.encrypt(&plaintext);
        lfsr.reset();
        let decrypted = lfsr.decrypt(&ciphertext);
        assert_eq!(plaintext, decrypted.as_slice());
    }

    #[test]
    fn test_lfsr_gen() {
        let coefs = vec![1, 0, 1, 1];
        let state = vec![1, 0, 1, 0];
        let mut lfsr = Lfsr::new(coefs, state);
        let mut nexts = Vec::new();
        for _ in 0..10 {
            nexts.push(lfsr.next());
        }
        assert_eq!(nexts, vec![1, 0, 1, 0, 0, 0, 1, 1, 0, 1]);
    }

    #[test]
    fn test_lfsr_solve() {
        let coefs = 4;
        let output = vec![1, 0, 1, 0, 0, 0, 1, 1];
        let coefs = solve_for_coefs(coefs, &output);
        assert_eq!(coefs, vec![1, 0, 1, 1]);
    }

    #[test]
    fn test_lfsr_read_binary_string() {
        let s = "10101010";
        let v = read_binary_string(s);
        assert_eq!(v, vec![1, 0, 1, 0, 1, 0, 1, 0]);
    }

    #[test]
    fn test_lfsr_to_binary_string() {
        let v = vec![1, 0, 1, 0, 1, 0, 1, 0];
        let s = to_binary_string(&v);
        assert_eq!(s, "10101010");
    }
}
