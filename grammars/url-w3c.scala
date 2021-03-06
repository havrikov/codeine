// Translated from https://www.w3.org/Addressing/URL/5_BNF.html

Grammar(
  //  'prefixedurl := "url:" ~ 'url,
  'url := 'httpaddress | 'ftpaddress | 'newsaddress | 'nntpaddress | 'prosperoaddress | 'telnetaddress | 'gopheraddress | 'waisaddress | 'mailtoaddress,
  //  'scheme := 'ialpha,
  'httpaddress := "http://" ~ 'hostport ~ ("/" ~ 'path).? ~ ("?" ~ 'search).?,
  'ftpaddress := "ftp://" ~ 'login ~ "/" ~ 'path ~ (";" ~ 'ftptype).?,
  'newsaddress := "news:" ~ 'groupart,
  'nntpaddress := "nntp:" ~ 'group ~ "/" ~ 'digits,
  'mailtoaddress := "mailto:" ~ 'xalphas ~ "@" ~ 'hostname,
  'waisaddress := 'waisindex | 'waisdoc,
  'waisindex := "wais://" ~ 'hostport ~ "/" ~ 'database ~ ("?" ~ 'search).?,
  'waisdoc := "wais://" ~ 'hostport ~ "/" ~ 'database ~ "/" ~ 'wtype ~ "/" ~ 'wpath,
  'wpath := ('digits ~ "=" ~ 'path ~ ";").rep(1),
  'groupart := "*" | 'group | 'article,
  'group := 'ialpha ~ ("." ~ 'ialpha).rep,
  'article := 'xalphas ~ "@" ~ 'host,
  'database := 'xalphas,
  'wtype := 'xalphas,
  'prosperoaddress := 'prosperolink,
  'prosperolink := "prospero://" ~ 'hostport ~ "/" ~ 'hsoname ~ ("%00" ~ 'version ~ 'attributes.?).?,
  'hsoname := 'path,
  'version := 'digits,
  'attributes := 'attribute.rep(1),
  'attribute := 'alphanums,
  'telnetaddress := "telnet://" ~ 'login,
  'gopheraddress := "gopher://" ~ 'hostport ~ ("/" ~ 'gtype ~ 'gcommand).?,
  'login := ('user ~ (":" ~ 'password).? ~ "@").? ~ 'hostport,
  'hostport := 'host ~ (":" ~ 'port).?,
  'host := 'hostname | 'hostnumber,
  'ftptype := "A" ~ 'formcode | "E" ~ 'formcode | "I" | "L" ~ 'digits,
  'formcode := "N" | "T" | "C",
  'hostname := 'ialpha ~ ("." ~ 'ialpha).rep,
  'hostnumber := 'digits ~ "." ~ 'digits ~ "." ~ 'digits ~ "." ~ 'digits,
  'port := 'digits,
  'gcommand := 'path,
  'path := ('segment ~ "/").rep ~ 'segment.?,
  'segment := 'xpalphas,
  'search := 'xalphas ~ ("+" ~ 'xalphas).rep,
  'user := 'alphanum2.rep(1),
  'password := 'alphanum2.rep(1),
  'gtype := 'xalpha,
  'alphanum2 := 'alpha | 'digit | "-" | "_" | "." | "+",
  'xalpha := 'alpha | 'digit | 'safe | 'extra | 'escape,
  'xalphas := 'xalpha.rep(1),
  'xpalpha := 'xalpha | "+",
  'xpalphas := 'xpalpha.rep(1),
  'ialpha := 'alpha ~ 'xalphas.?,
  'alpha := "[a-zA-Z]".regex,
  'digit := "[0-9]".regex,
  'safe := "$" | "_" | "@" | "." | "&" | "+" | "-",
  'extra := "!" | "*" | "\"" | "'" | "(" | ")" | ",",
  //  'reserved := "=" | ";" | "/" | "#" | "?" | ":" | " ",
  'escape := "%" ~ 'hex ~ 'hex,
  'hex := "[0-9a-fA-F]".regex,
  //  'national := "{" | "}" | "|" | "[" | "]" | "\\" | "^" | "~",
  //  'punctuation := "<" | ">",
  'digits := 'digit.rep(1),
  'alphanum := 'alpha | 'digit,
  'alphanums := 'alphanum.rep(1)
)
