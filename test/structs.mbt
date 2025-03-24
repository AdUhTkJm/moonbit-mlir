struct Struct {
  x : Int
  y : Int
};

fn Struct::new() -> Struct {
  { x : 0, y : 0 }
}

fn main {
  let s = Struct::new();
  let _ = s.x + s.y
}
