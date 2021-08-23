// Translated from https://json.org
// As of 2020-05-08

Grammar(
  'json := 'element,

  'value := 'object
    | 'array
    | 'string
    | 'number
    | "true"
    | "false"
    | "null"
  ,

  'object := "{" ~ 'ws ~ "}"
    | "{" ~ 'members ~ "}",

  'members := 'member ~ ("," ~ 'member).rep,

  'member := 'ws ~ 'string ~ 'ws ~ ":" ~ 'element,

  'array := "[" ~ 'ws ~ "]"
    | "[" ~ 'elements ~ "]",

  'elements := 'element ~ ("," ~ 'element).rep,

  'element := 'ws ~ 'value ~ 'ws,

  'string := "\"" ~ 'characters ~ "\"",

  'characters := 'character.rep,


  'character :=
    """[^\"\\\\]""".regex // anything but " and \
      | "\\" ~ 'escape,

  'escape :=
    """"""" // a "
      | """\""" // a \
      | "/"
      | "b"
      | "f"
      | "n"
      | "r"
      | "t"
      | "u" ~ 'hex ~ 'hex ~ 'hex,

  'hex := 'digit
    | "[A-F]".regex
    | "[a-f]".regex,


  'number := 'integer ~ 'fraction ~ 'exponent,

  'integer := 'digit
    | 'onenine ~ 'digits
    | "-" ~ 'digit
    | "-" ~ 'onenine ~ 'digits,

  'digits := 'digit.rep(1),

  'digit := "0"
    | 'onenine,

  'onenine := "[1-9]".regex,

  'fraction := ""
    | "." ~ 'digits,

  'exponent := ""
    | "E" ~ 'sign ~ 'digits
    | "e" ~ 'sign ~ 'digits,

  'sign := ""
    | "+"
    | "-",

  'ws := (" " | "\n" | "\r" | "\t").rep
)
