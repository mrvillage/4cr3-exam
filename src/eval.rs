use crate::{modinv, modpow, modulus};
use std::{iter::Peekable, vec::IntoIter};

#[derive(Debug, PartialEq, Eq)]
enum Token {
    Num(i128),
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    GroupStart,
    GroupEnd,
}

#[derive(Debug, PartialEq, Eq)]
enum NumExpr {
    Num(i128),
    Group(Box<Expr>),
}

impl NumExpr {
    pub fn parse(tokens: &mut Peekable<IntoIter<Token>>) -> Result<NumExpr, String> {
        match tokens.next() {
            Some(Token::Num(num)) => Ok(NumExpr::Num(num)),
            Some(Token::GroupStart) => {
                let expr = Expr::parse(tokens)?;
                match tokens.next() {
                    Some(Token::GroupEnd) => Ok(NumExpr::Group(Box::new(expr))),
                    _ => Err("Expected ')'".to_string()),
                }
            }
            _ => Err("Expected number or '('".to_string()),
        }
    }

    pub fn eval(self, m: i128) -> i128 {
        match self {
            NumExpr::Num(num) => modulus(num, m),
            NumExpr::Group(expr) => expr.0.eval(m),
        }
    }
}

impl std::fmt::Display for NumExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            NumExpr::Num(num) => write!(f, "{}", num),
            NumExpr::Group(expr) => write!(f, "({})", expr.0),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum PowExpr {
    Num(NumExpr),
    Pow(Box<PowExpr>, NumExpr),
}

impl PowExpr {
    pub fn parse(tokens: &mut Peekable<IntoIter<Token>>) -> Result<PowExpr, String> {
        let num_expr = NumExpr::parse(tokens)?;
        let mut pow_expr = PowExpr::Num(num_expr);
        while let Some(token) = tokens.peek() {
            match token {
                Token::Pow => {
                    tokens.next();
                    let num_expr = NumExpr::parse(tokens)?;
                    pow_expr = PowExpr::Pow(Box::new(pow_expr), num_expr);
                }
                _ => break,
            }
        }
        Ok(pow_expr)
    }

    pub fn eval(self, m: i128) -> i128 {
        match self {
            PowExpr::Num(num_expr) => num_expr.eval(m),
            PowExpr::Pow(lhs, rhs) => {
                let lhs = lhs.eval(m);
                let rhs = rhs.eval(m);
                modpow(lhs, rhs, m)
            }
        }
    }
}

impl std::fmt::Display for PowExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            PowExpr::Num(num_expr) => write!(f, "{}", num_expr),
            PowExpr::Pow(lhs, rhs) => write!(f, "({}^{})", lhs, rhs),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum MulExpr {
    Pow(PowExpr),
    Mul(Box<MulExpr>, PowExpr),
    Div(Box<MulExpr>, PowExpr),
}

impl MulExpr {
    pub fn parse(tokens: &mut Peekable<IntoIter<Token>>) -> Result<MulExpr, String> {
        let pow_expr = PowExpr::parse(tokens)?;
        let mut mul_expr = MulExpr::Pow(pow_expr);
        while let Some(token) = tokens.peek() {
            match token {
                Token::Mul => {
                    tokens.next();
                    let pow_expr = PowExpr::parse(tokens)?;
                    mul_expr = MulExpr::Mul(Box::new(mul_expr), pow_expr);
                }
                Token::Div => {
                    tokens.next();
                    let pow_expr = PowExpr::parse(tokens)?;
                    mul_expr = MulExpr::Div(Box::new(mul_expr), pow_expr);
                }
                _ => break,
            }
        }
        Ok(mul_expr)
    }

    pub fn eval(self, m: i128) -> i128 {
        match self {
            MulExpr::Pow(pow_expr) => pow_expr.eval(m),
            MulExpr::Mul(lhs, rhs) => modulus(lhs.eval(m) * rhs.eval(m), m),
            MulExpr::Div(lhs, rhs) => {
                let rhs = rhs.eval(m);
                let inv = modinv(rhs, m);
                modulus(lhs.eval(m) * inv, m)
            }
        }
    }
}

impl std::fmt::Display for MulExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            MulExpr::Pow(pow_expr) => write!(f, "{}", pow_expr),
            MulExpr::Mul(lhs, rhs) => write!(f, "({} * {})", lhs, rhs),
            MulExpr::Div(lhs, rhs) => write!(f, "({} / {})", lhs, rhs),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum AddExpr {
    Mul(MulExpr),
    Add(Box<AddExpr>, MulExpr),
    Sub(Box<AddExpr>, MulExpr),
}

impl AddExpr {
    pub fn parse(tokens: &mut Peekable<IntoIter<Token>>) -> Result<AddExpr, String> {
        let mul_expr = MulExpr::parse(tokens)?;
        let mut add_expr = AddExpr::Mul(mul_expr);
        while let Some(token) = tokens.peek() {
            match token {
                Token::Add => {
                    tokens.next();
                    let mul_expr = MulExpr::parse(tokens)?;
                    add_expr = AddExpr::Add(Box::new(add_expr), mul_expr);
                }
                Token::Sub => {
                    tokens.next();
                    let mul_expr = MulExpr::parse(tokens)?;
                    add_expr = AddExpr::Sub(Box::new(add_expr), mul_expr);
                }
                _ => break,
            }
        }
        Ok(add_expr)
    }

    pub fn eval(self, m: i128) -> i128 {
        match self {
            AddExpr::Mul(mul_expr) => mul_expr.eval(m),
            AddExpr::Add(lhs, rhs) => modulus(lhs.eval(m) + rhs.eval(m), m),
            AddExpr::Sub(lhs, rhs) => modulus(lhs.eval(m) - rhs.eval(m), m),
        }
    }
}

impl std::fmt::Display for AddExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            AddExpr::Mul(mul_expr) => write!(f, "{}", mul_expr),
            AddExpr::Add(lhs, rhs) => write!(f, "({} + {})", lhs, rhs),
            AddExpr::Sub(lhs, rhs) => write!(f, "({} - {})", lhs, rhs),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct Expr(AddExpr);

impl Expr {
    fn tokenize(input: &str) -> Result<Vec<Token>, String> {
        // Tokenize input
        let mut num = String::new();
        let mut tokens = Vec::new();
        let mut input = input.chars().filter(|c| !c.is_whitespace()).peekable();
        while let Some(c) = input.next() {
            match c {
                '0'..='9' => num.push(c),
                '-' if num.is_empty() && matches!(input.peek(), Some('0'..='9')) => num.push(c),
                '+' | '-' | '*' | '/' | '(' | ')' | '^' => {
                    if !num.is_empty() {
                        tokens.push(Token::Num(num.parse().unwrap()));
                        num.clear();
                    }
                    match c {
                        '+' => tokens.push(Token::Add),
                        '-' => tokens.push(Token::Sub),
                        '*' => tokens.push(Token::Mul),
                        '/' => tokens.push(Token::Div),
                        '^' => tokens.push(Token::Pow),
                        '(' => tokens.push(Token::GroupStart),
                        ')' => tokens.push(Token::GroupEnd),
                        _ => unreachable!(),
                    }
                }
                ' ' => (),
                _ => return Err(format!("Unexpected character '{}'", c)),
            }
        }
        if !num.is_empty() {
            tokens.push(Token::Num(num.parse().unwrap()));
        }

        Ok(tokens)
    }

    fn parse(tokens: &mut Peekable<IntoIter<Token>>) -> Result<Expr, String> {
        let add_expr = AddExpr::parse(tokens)?;
        Ok(Expr(add_expr))
    }

    pub fn eval(self, m: i128) -> i128 {
        self.0.eval(m)
    }

    pub fn parse_str(input: &str) -> Result<Expr, String> {
        let tokens = Expr::tokenize(input)?;
        let mut tokens = tokens.into_iter().peekable();
        Expr::parse(&mut tokens)
    }

    #[allow(dead_code)]
    pub fn eval_str(input: &str, m: i128) -> Result<i128, String> {
        let tokens = Expr::tokenize(input)?;
        let mut tokens = tokens.into_iter().peekable();
        let expr = Expr::parse(&mut tokens)?;
        if tokens.next().is_some() {
            return Err("Unexpected tokens".to_string());
        }
        Ok(expr.eval(m))
    }
}

impl std::fmt::Display for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut s = format!("{}", self.0);
        if s.starts_with('(') && s.ends_with(')') {
            s.pop();
            s.remove(0);
        }
        write!(f, "{}", s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize() {
        assert_eq!(
            Expr::tokenize("1 + 2 * (3 - 4) / 5").unwrap(),
            vec![
                Token::Num(1),
                Token::Add,
                Token::Num(2),
                Token::Mul,
                Token::GroupStart,
                Token::Num(3),
                Token::Sub,
                Token::Num(4),
                Token::GroupEnd,
                Token::Div,
                Token::Num(5),
            ]
        );
    }

    #[test]
    fn test_tokenize_negative() {
        assert_eq!(
            Expr::tokenize("1 + -2 * (3 - 4) / 5").unwrap(),
            vec![
                Token::Num(1),
                Token::Add,
                Token::Num(-2),
                Token::Mul,
                Token::GroupStart,
                Token::Num(3),
                Token::Sub,
                Token::Num(4),
                Token::GroupEnd,
                Token::Div,
                Token::Num(5),
            ]
        );
    }

    #[test]
    fn test_parse_num_expr() {
        let mut tokens = vec![Token::Num(42)].into_iter().peekable();
        assert_eq!(NumExpr::parse(&mut tokens).unwrap(), NumExpr::Num(42));
    }

    #[test]
    fn test_parse_group_expr() {
        let mut tokens = vec![Token::GroupStart, Token::Num(42), Token::GroupEnd]
            .into_iter()
            .peekable();
        assert_eq!(
            NumExpr::parse(&mut tokens).unwrap(),
            NumExpr::Group(Box::new(Expr(AddExpr::Mul(MulExpr::Pow(PowExpr::Num(
                NumExpr::Num(42)
            ))))))
        );
    }

    #[test]
    fn test_parse_pow_expr_num() {
        let mut tokens = vec![Token::Num(42)].into_iter().peekable();
        assert_eq!(
            PowExpr::parse(&mut tokens).unwrap(),
            PowExpr::Num(NumExpr::Num(42))
        );
    }

    #[test]
    fn test_parse_pow_expr_pow() {
        let mut tokens = vec![
            Token::Num(1),
            Token::Pow,
            Token::Num(2),
            Token::Pow,
            Token::Num(3),
        ]
        .into_iter()
        .peekable();
        assert_eq!(
            PowExpr::parse(&mut tokens).unwrap(),
            PowExpr::Pow(
                Box::new(PowExpr::Pow(
                    Box::new(PowExpr::Num(NumExpr::Num(1))),
                    NumExpr::Num(2)
                )),
                NumExpr::Num(3)
            )
        );
    }

    #[test]
    fn test_parse_mul_expr_num() {
        let mut tokens = vec![Token::Num(42)].into_iter().peekable();
        assert_eq!(
            MulExpr::parse(&mut tokens).unwrap(),
            MulExpr::Pow(PowExpr::Num(NumExpr::Num(42)))
        );
    }

    #[test]
    fn test_parse_mul_expr_mul() {
        let mut tokens = vec![
            Token::Num(1),
            Token::Mul,
            Token::Num(2),
            Token::Mul,
            Token::Num(3),
        ]
        .into_iter()
        .peekable();
        assert_eq!(
            MulExpr::parse(&mut tokens).unwrap(),
            MulExpr::Mul(
                Box::new(MulExpr::Mul(
                    Box::new(MulExpr::Pow(PowExpr::Num(NumExpr::Num(1)))),
                    PowExpr::Num(NumExpr::Num(2))
                )),
                PowExpr::Num(NumExpr::Num(3))
            )
        );
    }

    #[test]
    fn test_parse_mul_expr_div() {
        let mut tokens = vec![
            Token::Num(1),
            Token::Div,
            Token::Num(2),
            Token::Div,
            Token::Num(3),
        ]
        .into_iter()
        .peekable();
        assert_eq!(
            MulExpr::parse(&mut tokens).unwrap(),
            MulExpr::Div(
                Box::new(MulExpr::Div(
                    Box::new(MulExpr::Pow(PowExpr::Num(NumExpr::Num(1)))),
                    PowExpr::Num(NumExpr::Num(2))
                )),
                PowExpr::Num(NumExpr::Num(3))
            )
        );
    }

    #[test]
    fn test_parse_add_expr_mul() {
        let mut tokens = vec![Token::Num(42)].into_iter().peekable();
        assert_eq!(
            AddExpr::parse(&mut tokens).unwrap(),
            AddExpr::Mul(MulExpr::Pow(PowExpr::Num(NumExpr::Num(42))))
        );
    }

    #[test]
    fn test_parse_add_expr_add() {
        let mut tokens = vec![
            Token::Num(1),
            Token::Add,
            Token::Num(2),
            Token::Add,
            Token::Num(3),
        ]
        .into_iter()
        .peekable();
        assert_eq!(
            AddExpr::parse(&mut tokens).unwrap(),
            AddExpr::Add(
                Box::new(AddExpr::Add(
                    Box::new(AddExpr::Mul(MulExpr::Pow(PowExpr::Num(NumExpr::Num(1))))),
                    MulExpr::Pow(PowExpr::Num(NumExpr::Num(2)))
                )),
                MulExpr::Pow(PowExpr::Num(NumExpr::Num(3)))
            )
        );
    }

    #[test]
    fn test_parse_add_expr_sub() {
        let mut tokens = vec![
            Token::Num(1),
            Token::Sub,
            Token::Num(2),
            Token::Sub,
            Token::Num(3),
        ]
        .into_iter()
        .peekable();
        assert_eq!(
            AddExpr::parse(&mut tokens).unwrap(),
            AddExpr::Sub(
                Box::new(AddExpr::Sub(
                    Box::new(AddExpr::Mul(MulExpr::Pow(PowExpr::Num(NumExpr::Num(1))))),
                    MulExpr::Pow(PowExpr::Num(NumExpr::Num(2)))
                )),
                MulExpr::Pow(PowExpr::Num(NumExpr::Num(3)))
            )
        );
    }

    #[test]
    fn test_parse_expr() {
        // 1 + 2 * (3 - 4) / 5
        let mut tokens = vec![
            Token::Num(1),
            Token::Add,
            Token::Num(2),
            Token::Mul,
            Token::GroupStart,
            Token::Num(3),
            Token::Sub,
            Token::Num(4),
            Token::GroupEnd,
            Token::Div,
            Token::Num(5),
        ]
        .into_iter()
        .peekable();
        assert_eq!(
            Expr::parse(&mut tokens).unwrap(),
            Expr(AddExpr::Add(
                Box::new(AddExpr::Mul(MulExpr::Pow(PowExpr::Num(NumExpr::Num(1))))),
                MulExpr::Div(
                    Box::new(MulExpr::Mul(
                        Box::new(MulExpr::Pow(PowExpr::Num(NumExpr::Num(2)))),
                        PowExpr::Num(NumExpr::Group(Box::new(Expr(AddExpr::Sub(
                            Box::new(AddExpr::Mul(MulExpr::Pow(PowExpr::Num(NumExpr::Num(3))))),
                            MulExpr::Pow(PowExpr::Num(NumExpr::Num(4)))
                        )))))
                    )),
                    PowExpr::Num(NumExpr::Num(5))
                )
            ))
        );
    }

    #[test]
    fn test_eval_num_expr() {
        let num_expr = NumExpr::Num(42);
        assert_eq!(num_expr.eval(10), 2);
    }

    #[test]
    fn test_eval_num_expr_group() {
        let num_expr = NumExpr::Group(Box::new(Expr(AddExpr::Mul(MulExpr::Pow(PowExpr::Num(
            NumExpr::Num(42),
        ))))));
        assert_eq!(num_expr.eval(10), 2);
    }

    #[test]
    fn test_eval_pow_expr_num() {
        let pow_expr = PowExpr::Num(NumExpr::Num(42));
        assert_eq!(pow_expr.eval(10), 2);
    }

    #[test]
    fn test_eval_pow_expr_pow() {
        // 1 ^ 2 ^ 3 = 1
        let pow_expr = PowExpr::Pow(
            Box::new(PowExpr::Pow(
                Box::new(PowExpr::Num(NumExpr::Num(1))),
                NumExpr::Num(2),
            )),
            NumExpr::Num(3),
        );
        assert_eq!(pow_expr.eval(10), 1);
    }

    #[test]
    fn test_eval_mul_expr_num() {
        let mul_expr = MulExpr::Pow(PowExpr::Num(NumExpr::Num(42)));
        assert_eq!(mul_expr.eval(10), 2);
    }

    #[test]
    fn test_eval_mul_expr_mul() {
        let mul_expr = MulExpr::Mul(
            Box::new(MulExpr::Mul(
                Box::new(MulExpr::Pow(PowExpr::Num(NumExpr::Num(1)))),
                PowExpr::Num(NumExpr::Num(2)),
            )),
            PowExpr::Num(NumExpr::Num(3)),
        );
        assert_eq!(mul_expr.eval(10), 6);
    }

    #[test]
    fn test_eval_mul_expr_div() {
        // 1 / 2 / 3 = 4
        let mul_expr = MulExpr::Div(
            Box::new(MulExpr::Div(
                Box::new(MulExpr::Pow(PowExpr::Num(NumExpr::Num(1)))),
                PowExpr::Num(NumExpr::Num(2)),
            )),
            PowExpr::Num(NumExpr::Num(3)),
        );
        assert_eq!(mul_expr.eval(10), 4);
    }

    #[test]
    fn test_eval_add_expr_mul() {
        let add_expr = AddExpr::Mul(MulExpr::Pow(PowExpr::Num(NumExpr::Num(1))));
        assert_eq!(add_expr.eval(10), 1);
    }

    #[test]
    fn test_eval_add_expr_add() {
        let add_expr = AddExpr::Add(
            Box::new(AddExpr::Add(
                Box::new(AddExpr::Mul(MulExpr::Pow(PowExpr::Num(NumExpr::Num(1))))),
                MulExpr::Pow(PowExpr::Num(NumExpr::Num(2))),
            )),
            MulExpr::Pow(PowExpr::Num(NumExpr::Num(3))),
        );
        assert_eq!(add_expr.eval(10), 6);
    }

    #[test]
    fn test_eval_add_expr_sub() {
        // 1 - 2 - 3 = 6
        let add_expr = AddExpr::Sub(
            Box::new(AddExpr::Sub(
                Box::new(AddExpr::Mul(MulExpr::Pow(PowExpr::Num(NumExpr::Num(1))))),
                MulExpr::Pow(PowExpr::Num(NumExpr::Num(2))),
            )),
            MulExpr::Pow(PowExpr::Num(NumExpr::Num(3))),
        );
        assert_eq!(add_expr.eval(10), 6);
    }

    #[test]
    fn test_eval_expr() {
        // 1 + 2 * (3 - 4) / 5
        let expr = Expr(AddExpr::Add(
            Box::new(AddExpr::Mul(MulExpr::Pow(PowExpr::Num(NumExpr::Num(1))))),
            MulExpr::Div(
                Box::new(MulExpr::Mul(
                    Box::new(MulExpr::Pow(PowExpr::Num(NumExpr::Num(2)))),
                    PowExpr::Num(NumExpr::Group(Box::new(Expr(AddExpr::Sub(
                        Box::new(AddExpr::Mul(MulExpr::Pow(PowExpr::Num(NumExpr::Num(3))))),
                        MulExpr::Pow(PowExpr::Num(NumExpr::Num(4))),
                    ))))),
                )),
                PowExpr::Num(NumExpr::Num(5)),
            ),
        ));
        assert_eq!(expr.eval(10), 1);
    }

    #[test]
    fn test_eval_str_1() {
        assert_eq!(Expr::eval_str("1 + 2 * (3 - 4) / 5", 10).unwrap(), 1);
    }

    #[test]
    fn test_eval_str_2() {
        assert_eq!(Expr::eval_str("1 + 2 * (3 - 4) / 5", 100).unwrap(), 91);
    }

    #[test]
    fn test_eval_str_3() {
        assert_eq!(Expr::eval_str("(7 + 3) * 2", 10).unwrap(), 0);
    }

    #[test]
    fn test_eval_str_4() {
        assert_eq!(Expr::eval_str("(-7 + 3) * 2", 10).unwrap(), 2);
    }

    #[test]
    fn test_eval_str_5() {
        assert_eq!(Expr::eval_str("(-7 + 3) * 2", 100).unwrap(), 92);
    }

    #[test]
    fn test_eval_str_6() {
        assert_eq!(Expr::eval_str("(-7 + 3)^2 * 2", 10).unwrap(), 2);
    }
}
