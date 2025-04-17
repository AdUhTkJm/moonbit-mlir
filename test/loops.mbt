fn main {
  let mut x = 19;
  let mut count = 0;
  while (x != 1) {
    if (x % 2 == 0) {
      x /= 2;
    } else {
      x = x * 3 + 1;
    }
  }
}
