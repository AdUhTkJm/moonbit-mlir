fn fib(x: Int) -> Int {
  if (x < 1) {
    0
  } else {
    fib(x - 1) + fib(x - 2)
  }
}

fn main {
  let _ = fib(7)
}
