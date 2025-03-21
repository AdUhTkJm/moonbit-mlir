fn print_d(s : Int) -> Unit = "%println"

fn empty() -> Unit {
  
}

fn func(x: Int) -> Int {
  x;
}

fn main {
  let x = 1 + 2 * 3
  let y = 2
  let z = x / y
  let w = if z > 0 { x } else { y }
  print_d(func(w))
}
