Grammar(
  'START := 'Script,
  'ClassElement := ((";") /* ClassElement//0 */ | ('MethodDefinition) /* ClassElement//1 */ | ((" static " ~ 'WhiteSpace ~ 'MethodDefinition)) /* ClassElement//2 */),
  'ObjectLiteral := ((("{" ~ "}")) /* ObjectLiteral//0 */ | (("{" ~ 'PropertyDefinitionList ~ "," ~ "}")) /* ObjectLiteral//1 */ | (("{" ~ 'PropertyDefinitionList ~ "}")) /* ObjectLiteral//2 */),
  'UnaryExpression := (('PostfixExpression) /* UnaryExpression//0 */ | (("!" ~ 'UnaryExpression)) /* UnaryExpression//1 */ | (("+" ~ 'UnaryExpression)) /* UnaryExpression//2 */ | (("++" ~ 'UnaryExpression)) /* UnaryExpression//3 */ | (("-" ~ 'UnaryExpression)) /* UnaryExpression//4 */ | (("--" ~ 'UnaryExpression)) /* UnaryExpression//5 */ | ((" delete " ~ 'UnaryExpression)) /* UnaryExpression//6 */ | ((" typeof " ~ 'UnaryExpression)) /* UnaryExpression//7 */ | ((" void " ~ 'UnaryExpression)) /* UnaryExpression//8 */ | (("~" ~ 'UnaryExpression)) /* UnaryExpression//9 */),
  'Term := (('Assertion) /* Term//0 */ | ('Atom) /* Term//1 */ | (('Atom ~ 'Quantifier)) /* Term//2 */),
  'ArrowFormalParameters := ("(" ~ 'StrictFormalParameters ~ ")"),
  'SignedInteger := (('DecimalDigits) /* SignedInteger//0 */ | (("+" ~ 'DecimalDigits)) /* SignedInteger//1 */ | (("-" ~ 'DecimalDigits)) /* SignedInteger//2 */),
  'DoubleStringCharacter := (('LineContinuation) /* DoubleStringCharacter//0 */ | (("\\" ~ 'EscapeSequence)) /* DoubleStringCharacter//1 */ | ("[^\\\"|\"\\\\\"|\r|\n]".regex) /* DoubleStringCharacter//2 */),
  'PatternCharacter := "[^$\\.^*+?)(][}{|]".regex,
  'Atom := ((".") /* Atom//0 */ | ('CharacterClass) /* Atom//1 */ | ('PatternCharacter) /* Atom//2 */ | (("(" ~ "?" ~ ":" ~ 'Disjunction ~ ")")) /* Atom//3 */ | (("(" ~ 'Disjunction ~ ")")) /* Atom//4 */ | (("\\" ~ 'AtomEscape)) /* Atom//5 */),
  'NonemptyClassRangesNoDash := (('ClassAtom) /* NonemptyClassRangesNoDash//0 */ | (('ClassAtomNoDash ~ "-" ~ 'ClassAtom ~ 'ClassRanges)) /* NonemptyClassRangesNoDash//1 */ | (('ClassAtomNoDash ~ 'NonemptyClassRangesNoDash)) /* NonemptyClassRangesNoDash//2 */),
  'ScriptBody := 'StatementList,
  'NonEscapeCharacter := "[^'\"\\bfnrtvxu\r\n]".regex,
  'DefaultClause := (" default " ~ ":" ~ ('StatementList).?),
  'DoubleStringCharacters := ('DoubleStringCharacter).rep(1),
  'HexDigits := ('HexDigit).rep(1),
  'PropertyName := (('ComputedPropertyName) /* PropertyName//0 */ | ('LiteralPropertyName) /* PropertyName//1 */),
  'HT := "\t",
  'EqualityExpression := (('RelationalExpression) /* EqualityExpression//0 */ | (('EqualityExpression ~ "!=" ~ 'RelationalExpression)) /* EqualityExpression//1 */ | (('EqualityExpression ~ "!==" ~ 'RelationalExpression)) /* EqualityExpression//2 */ | (('EqualityExpression ~ "==" ~ 'RelationalExpression)) /* EqualityExpression//3 */ | (('EqualityExpression ~ "===" ~ 'RelationalExpression)) /* EqualityExpression//4 */),
  'MultiplicativeExpression := (('UnaryExpression) /* MultiplicativeExpression//0 */ | (('MultiplicativeExpression ~ 'MultiplicativeOperator ~ 'UnaryExpression)) /* MultiplicativeExpression//1 */),
  'ShiftExpression := (('AdditiveExpression) /* ShiftExpression//0 */ | (('ShiftExpression ~ "<<" ~ 'AdditiveExpression)) /* ShiftExpression//1 */ | (('ShiftExpression ~ ">>" ~ 'AdditiveExpression ~ 'ShiftExpression ~ ">>>" ~ 'AdditiveExpression)) /* ShiftExpression//2 */),
  'LineTerminatorSequence := 'LF,
  'DecimalDigits := (('DecimalDigit) /* DecimalDigits//0 */ | (('DecimalDigits ~ 'DecimalDigit)) /* DecimalDigits//1 */),
  'ExportClause := ((("{" ~ "}")) /* ExportClause//0 */ | (("{" ~ 'ExportsList ~ "," ~ "}")) /* ExportClause//1 */ | (("{" ~ 'ExportsList ~ "}")) /* ExportClause//2 */),
  'AtomEscape := (('CharacterClassEscape) /* AtomEscape//0 */ | ('CharacterEscape) /* AtomEscape//1 */ | ('DecimalEscape) /* AtomEscape//2 */),
  'FormalParameters := (("") /* FormalParameters//0 */ | ('FormalParameterList) /* FormalParameters//1 */),
  'Pattern := 'Disjunction,
  'CallExpression := (('SuperCall) /* CallExpression//0 */ | (('CallExpression ~ "." ~ 'IdentifierName)) /* CallExpression//1 */ | (('CallExpression ~ "[" ~ 'Expression ~ "]")) /* CallExpression//2 */ | (('CallExpression ~ 'Arguments)) /* CallExpression//3 */ | (('CallExpression ~ 'TemplateLiteral)) /* CallExpression//4 */ | (('MemberExpression ~ 'Arguments)) /* CallExpression//5 */),
  'IterationStatement := (((" do " ~ 'Statement ~ " while " ~ "(" ~ 'Expression ~ ")" ~ ";")) /* IterationStatement//0 */ | ((" for " ~ "(" ~ " var " ~ 'WhiteSpace ~ 'ForBinding ~ " in " ~ 'Expression ~ ")" ~ 'Statement)) /* IterationStatement//1 */ | ((" for " ~ "(" ~ " var " ~ 'WhiteSpace ~ 'ForBinding ~ " of " ~ 'AssignmentExpression ~ ")" ~ 'Statement)) /* IterationStatement//2 */ | ((" for " ~ "(" ~ " var " ~ 'WhiteSpace ~ 'VariableDeclarationList ~ ";" ~ ('Expression).? ~ ";" ~ ('Expression).? ~ ")" ~ 'Statement)) /* IterationStatement//3 */ | ((" for " ~ "(" ~ 'ForDeclaration ~ " in " ~ 'Expression ~ ")" ~ 'Statement)) /* IterationStatement//4 */ | ((" for " ~ "(" ~ 'ForDeclaration ~ " of " ~ 'AssignmentExpression ~ ")" ~ 'Statement)) /* IterationStatement//5 */ | ((" for " ~ "(" ~ 'LeftHandSideExpression ~ " in " ~ 'Expression ~ ")" ~ 'Statement)) /* IterationStatement//6 */ | ((" for " ~ "(" ~ 'LeftHandSideExpression ~ " of " ~ 'AssignmentExpression ~ ")" ~ 'Statement)) /* IterationStatement//7 */ | ((" for " ~ "(" ~ 'LexicalDeclaration ~ ('Expression).? ~ ";" ~ ('Expression).? ~ ")" ~ 'Statement)) /* IterationStatement//8 */ | ((" for " ~ "(" ~ ('Expression).? ~ ";" ~ ('Expression).? ~ ";" ~ ('Expression).? ~ ")" ~ 'Statement)) /* IterationStatement//9 */ | ((" while " ~ "(" ~ 'Expression ~ ")" ~ 'Statement)) /* IterationStatement//10 */),
  'Element := (((('Elision).? ~ 'AssignmentExpression)) /* Element//0 */ | ((('Elision).? ~ 'SpreadElement)) /* Element//1 */),
  'CaseClause := ("case " ~ 'Expression ~ ":" ~ ('StatementList).?),
  'StatementList := ((('StatementList ~ 'StatementListItem ~ ('LineTerminator).?)) /* StatementList//0 */ | (('StatementListItem ~ ('LineTerminator).?)) /* StatementList//1 */),
  'OctalDigits := (('OctalDigit) /* OctalDigits//0 */ | (('OctalDigits ~ 'OctalDigit)) /* OctalDigits//1 */),
  'ArrayLiteral := ((("[" ~ 'ElementList ~ "," ~ ('Elision).? ~ "]")) /* ArrayLiteral//0 */ | (("[" ~ 'ElementList ~ "]")) /* ArrayLiteral//1 */ | (("[" ~ ('Elision).? ~ "]")) /* ArrayLiteral//2 */),
  'RegExpUnicodeEscapeSequence := ((("u" ~ 'Hex4Digits)) /* RegExpUnicodeEscapeSequence//0 */ | (("u" ~ 'LeadSurrogate ~ "\\u" ~ 'TrailSurrogate)) /* RegExpUnicodeEscapeSequence//1 */ | (("u" ~ 'LeadSurrogate)) /* RegExpUnicodeEscapeSequence//2 */ | (("u" ~ 'NonSurrogate)) /* RegExpUnicodeEscapeSequence//3 */ | (("u" ~ 'TrailSurrogate)) /* RegExpUnicodeEscapeSequence//4 */ | (("u{" ~ 'HexDigits ~ "}")) /* RegExpUnicodeEscapeSequence//5 */),
  'ImportSpecifier := (('ImportedBinding) /* ImportSpecifier//0 */ | (('IdentifierName ~ 'WhiteSpace ~ " as " ~ 'WhiteSpace ~ 'ImportedBinding)) /* ImportSpecifier//1 */),
  'ImportedDefaultBinding := 'ImportedBinding,
  'BitwiseXORExpression := (('BitwiseANDExpression) /* BitwiseXORExpression//0 */ | (('BitwiseXORExpression ~ "^" ~ 'BitwiseANDExpression)) /* BitwiseXORExpression//1 */),
  'ComputedPropertyName := ("[" ~ 'AssignmentExpression ~ "]"),
  'IfStatement := (((" if " ~ "(" ~ 'Expression ~ ")" ~ 'Statement ~ " else " ~ 'Statement)) /* IfStatement//0 */ | ((" if " ~ "(" ~ 'Expression ~ ")" ~ 'Statement)) /* IfStatement//1 */),
  'BlockStatement := 'Block,
  'DecimalIntegerLiteral := (("0") /* DecimalIntegerLiteral//0 */ | (('NonZeroDigit ~ ('DecimalDigits).?)) /* DecimalIntegerLiteral//1 */),
  'BreakableStatement := (('IterationStatement) /* BreakableStatement//0 */ | ('SwitchStatement) /* BreakableStatement//1 */),
  'BinaryIntegerLiteral := ((("0B" ~ 'BinaryDigits)) /* BinaryIntegerLiteral//0 */ | (("0b" ~ 'BinaryDigits)) /* BinaryIntegerLiteral//1 */),
  'IdentifierReference := 'Identifier,
  'LeadSurrogate := 'Hex4Digits,
  'BindingPattern := (('ArrayBindingPattern) /* BindingPattern//0 */ | ('ObjectBindingPattern) /* BindingPattern//1 */),
  'PrimaryExpression := ((" this ") /* PrimaryExpression//0 */ | ('ArrayLiteral) /* PrimaryExpression//1 */ | ('ClassExpression) /* PrimaryExpression//2 */ | ('FunctionExpression) /* PrimaryExpression//3 */ | ('GeneratorExpression) /* PrimaryExpression//4 */ | ('IdentifierReference) /* PrimaryExpression//5 */ | ('Literal) /* PrimaryExpression//6 */ | ('ObjectLiteral) /* PrimaryExpression//7 */ | ('ParenthesizedExpression) /* PrimaryExpression//8 */ | ('RegularExpressionLiteral) /* PrimaryExpression//9 */ | ('TemplateLiteral) /* PrimaryExpression//10 */),
  'NewExpression := (('MemberExpression) /* NewExpression//0 */ | ((" new " ~ 'NewExpression)) /* NewExpression//1 */),
  'SubstitutionTemplate := ("`" ~ (('TemplateCharacters | 'TemplateInlay)).rep ~ "`"),
  'ExportsList := (('ExportSpecifier) /* ExportsList//0 */ | (('ExportsList ~ "," ~ 'ExportSpecifier)) /* ExportsList//1 */),
  'BindingElementList := (('BindingElisionElement) /* BindingElementList//0 */ | (('BindingElementList ~ "," ~ 'BindingElisionElement)) /* BindingElementList//1 */),
  'ParenthesizedExpression := ("(" ~ 'Expression ~ ")"),
  'AwaitExpression := (" await " ~ 'WhiteSpace ~ 'AssignmentExpression),
  'MetaProperty := 'NewTarget,
  'WhiteSpace := (('HT) /* WhiteSpace//0 */ | ('SP) /* WhiteSpace//1 */),
  'ArrayBindingPattern := ((("[" ~ 'BindingElementList ~ "," ~ ('Elision).? ~ ('BindingRestElement).? ~ "]")) /* ArrayBindingPattern//0 */ | (("[" ~ 'BindingElementList ~ "]")) /* ArrayBindingPattern//1 */ | (("[" ~ ('Elision).? ~ ('BindingRestElement).? ~ "]")) /* ArrayBindingPattern//2 */),
  'ClassBody := 'ClassElementList,
  'ContinueStatement := (((" continue " ~ ";")) /* ContinueStatement//0 */ | ((" continue " ~ 'WhiteSpace ~ 'LabelIdentifier ~ ";")) /* ContinueStatement//1 */),
  'CaseBlock := ((("{" ~ ('CaseClauses).? ~ "}")) /* CaseBlock//0 */ | (("{" ~ ('CaseClauses).? ~ 'DefaultClause ~ ('CaseClauses).? ~ "}")) /* CaseBlock//1 */),
  'ModuleSpecifier := 'StringLiteral,
  'StringLiteral := ((("'" ~ ('SingleStringCharacters).? ~ "'")) /* StringLiteral//0 */ | (("\"" ~ ('DoubleStringCharacters).? ~ "\"")) /* StringLiteral//1 */),
  'Literal := (('BooleanLiteral) /* Literal//0 */ | ('NullLiteral) /* Literal//1 */ | ('NumericLiteral) /* Literal//2 */ | ('StringLiteral) /* Literal//3 */),
  'NonSurrogate := 'Hex4Digits,
  'NamedImports := ((("{" ~ "}")) /* NamedImports//0 */ | (("{" ~ 'ImportsList ~ "," ~ "}")) /* NamedImports//1 */ | (("{" ~ 'ImportsList ~ "}")) /* NamedImports//2 */),
  'ClassAtom := (("-") /* ClassAtom//0 */ | ('ClassAtomNoDash) /* ClassAtom//1 */),
  'SP := " ",
  'FunctionRestParameter := 'BindingRestElement,
  'BindingProperty := (('SingleNameBinding) /* BindingProperty//0 */ | (('PropertyName ~ ":" ~ 'BindingElement)) /* BindingProperty//1 */),
  'BitwiseANDExpression := (('EqualityExpression) /* BitwiseANDExpression//0 */ | (('BitwiseANDExpression ~ "&" ~ 'EqualityExpression)) /* BitwiseANDExpression//1 */),
  'Expression := (('AssignmentExpression) /* Expression//0 */ | (('Expression ~ "," ~ 'AssignmentExpression)) /* Expression//1 */),
  'NonZeroDigit := (("1") /* NonZeroDigit//0 */ | ("2") /* NonZeroDigit//1 */ | ("3") /* NonZeroDigit//2 */ | ("4") /* NonZeroDigit//3 */ | ("5") /* NonZeroDigit//4 */ | ("6") /* NonZeroDigit//5 */ | ("7") /* NonZeroDigit//6 */ | ("8") /* NonZeroDigit//7 */ | ("9") /* NonZeroDigit//8 */),
  'LogicalORExpression := (('LogicalANDExpression) /* LogicalORExpression//0 */ | (('LogicalORExpression ~ "||" ~ 'LogicalANDExpression)) /* LogicalORExpression//1 */),
  'ArrowFunction := (((" async " ~ 'WhiteSpace)).? ~ 'ArrowParameters ~ 'WhiteSpace ~ "=>" ~ 'WhiteSpace ~ 'ConciseBody),
  'SingleStringCharacter := (('LineContinuation) /* SingleStringCharacter//0 */ | (("\\" ~ 'EscapeSequence)) /* SingleStringCharacter//1 */ | ("[^'|\\\\|\r|\n]".regex) /* SingleStringCharacter//2 */),
  'CatchParameter := (('BindingIdentifier) /* CatchParameter//0 */ | ('BindingPattern) /* CatchParameter//1 */),
  'BindingPropertyList := (('BindingProperty) /* BindingPropertyList//0 */ | (('BindingPropertyList ~ "," ~ 'BindingProperty)) /* BindingPropertyList//1 */),
  'QuantifierPrefix := (("*") /* QuantifierPrefix//0 */ | ("+") /* QuantifierPrefix//1 */ | ("?") /* QuantifierPrefix//2 */ | (("{" ~ 'DecimalDigits ~ "," ~ "}")) /* QuantifierPrefix//3 */ | (("{" ~ 'DecimalDigits ~ "," ~ 'DecimalDigits ~ "}")) /* QuantifierPrefix//4 */ | (("{" ~ 'DecimalDigits ~ "}")) /* QuantifierPrefix//5 */),
  'CoverInitializedName := ('IdentifierReference ~ 'WhiteSpace ~ 'Initializer),
  'CaseClauses := ((('CaseClause ~ 'LineTerminator)) /* CaseClauses//0 */ | (('CaseClauses ~ 'CaseClause)) /* CaseClauses//1 */),
  'SyntaxCharacter := (("$") /* SyntaxCharacter//0 */ | ("(") /* SyntaxCharacter//1 */ | (")") /* SyntaxCharacter//2 */ | ("*") /* SyntaxCharacter//3 */ | ("+") /* SyntaxCharacter//4 */ | (".") /* SyntaxCharacter//5 */ | ("?") /* SyntaxCharacter//6 */ | ("[") /* SyntaxCharacter//7 */ | ("\\") /* SyntaxCharacter//8 */ | ("]") /* SyntaxCharacter//9 */ | ("^") /* SyntaxCharacter//10 */ | ("{") /* SyntaxCharacter//11 */ | ("|") /* SyntaxCharacter//12 */ | ("}") /* SyntaxCharacter//13 */),
  'SingleNameBinding := ('BindingIdentifier ~ (('WhiteSpace ~ 'Initializer)).?),
  'BitwiseORExpression := (('BitwiseXORExpression) /* BitwiseORExpression//0 */ | (('BitwiseORExpression ~ "|" ~ 'BitwiseXORExpression)) /* BitwiseORExpression//1 */),
  'ConditionalExpression := (('LogicalORExpression) /* ConditionalExpression//0 */ | (('LogicalORExpression ~ "?" ~ 'AssignmentExpression ~ ":" ~ 'AssignmentExpression)) /* ConditionalExpression//1 */),
  'YieldExpression := (((" yield " ~ "*" ~ 'WhiteSpace ~ 'AssignmentExpression)) /* YieldExpression//0 */ | ((" yield " ~ 'WhiteSpace ~ 'AssignmentExpression)) /* YieldExpression//1 */),
  'AssignmentExpression := (('ArrowFunction) /* AssignmentExpression//0 */ | ('AwaitExpression) /* AssignmentExpression//1 */ | ('ConditionalExpression) /* AssignmentExpression//2 */ | ('YieldExpression) /* AssignmentExpression//3 */ | (('LeftHandSideExpression ~ "=" ~ 'AssignmentExpression)) /* AssignmentExpression//4 */ | (('LeftHandSideExpression ~ 'WhiteSpace ~ 'AssignmentOperator ~ 'WhiteSpace ~ 'AssignmentExpression)) /* AssignmentExpression//5 */),
  'GeneratorExpression := ("function*" ~ ('BindingIdentifier).? ~ "(" ~ 'FormalParameters ~ ") {" ~ 'GeneratorBody ~ "}"),
  'ArrowParameters := (('ArrowFormalParameters) /* ArrowParameters//0 */ | ('BindingIdentifier) /* ArrowParameters//1 */),
  'ImportClause := (('ImportedDefaultBinding) /* ImportClause//0 */ | ('NameSpaceImport) /* ImportClause//1 */ | ('NamedImports) /* ImportClause//2 */ | (('ImportedDefaultBinding ~ "," ~ 'NameSpaceImport)) /* ImportClause//3 */ | (('ImportedDefaultBinding ~ "," ~ 'NamedImports)) /* ImportClause//4 */),
  'ModuleDummy1 := (('Module) /* ModuleDummy1//0 */ | ('ModuleDummy2) /* ModuleDummy1//1 */),
  'Hex4Digits := ('HexDigit).rep(4, 4),
  'ModuleItemList := ((('ModuleItem ~ 'LineTerminator)) /* ModuleItemList//0 */ | (('ModuleItemList ~ 'ModuleItem)) /* ModuleItemList//1 */),
  'OctalDigit := (("0") /* OctalDigit//0 */ | ("1") /* OctalDigit//1 */ | ("2") /* OctalDigit//2 */ | ("3") /* OctalDigit//3 */ | ("4") /* OctalDigit//4 */ | ("5") /* OctalDigit//5 */ | ("6") /* OctalDigit//6 */ | ("7") /* OctalDigit//7 */),
  'ClassDeclaration := (" class " ~ 'WhiteSpace ~ 'BindingIdentifier ~ 'ClassTail),
  'NewTarget := ("new" ~ "." ~ "target"),
  'TemplateCharacter := (("$") /* TemplateCharacter//0 */ | ('LineContinuation) /* TemplateCharacter//1 */ | ('LineTerminatorSequence) /* TemplateCharacter//2 */ | (("\\" ~ 'EscapeSequence)) /* TemplateCharacter//3 */ | ("[^\n\u2028\r\u2029`$\\\\]".regex) /* TemplateCharacter//4 */),
  'MemberExpression := (('MetaProperty) /* MemberExpression//0 */ | ('PrimaryExpression) /* MemberExpression//1 */ | ('SuperProperty) /* MemberExpression//2 */ | ((" new " ~ 'WhiteSpace ~ 'MemberExpression ~ 'Arguments)) /* MemberExpression//3 */ | (('MemberExpression ~ "." ~ 'IdentifierName)) /* MemberExpression//4 */ | (('MemberExpression ~ "[" ~ 'Expression ~ "]")) /* MemberExpression//5 */ | (('MemberExpression ~ 'TemplateLiteral)) /* MemberExpression//6 */),
  'BindingElisionElement := (('Elision).? ~ 'BindingElement),
  'EmptyStatement := ";",
  'UnicodeEscapeSequence := ((("u" ~ 'Hex4Digits)) /* UnicodeEscapeSequence//0 */ | (("u{" ~ 'HexDigits ~ "}")) /* UnicodeEscapeSequence//1 */),
  'Quantifier := (('QuantifierPrefix) /* Quantifier//0 */ | (('QuantifierPrefix ~ "?")) /* Quantifier//1 */),
  'GeneratorMethod := ("*" ~ 'PropertyName ~ "(" ~ 'StrictFormalParameters ~ ") {" ~ 'GeneratorBody ~ "}"),
  'GeneratorDeclaration := ((("function*" ~ 'WhiteSpace ~ 'BindingIdentifier ~ "(" ~ 'FormalParameters ~ ") {" ~ 'GeneratorBody ~ "}")) /* GeneratorDeclaration//0 */ | (("function*(" ~ 'FormalParameters ~ ") {" ~ 'GeneratorBody ~ "}")) /* GeneratorDeclaration//1 */),
  'NameSpaceImport := ("* as" ~ 'WhiteSpace ~ 'ImportedBinding),
  'ImportDeclaration := (((" import " ~ 'WhiteSpace ~ 'ImportClause ~ 'WhiteSpace ~ 'FromClause ~ ";" ~ 'LineTerminator)) /* ImportDeclaration//0 */ | ((" import " ~ 'WhiteSpace ~ 'ModuleSpecifier ~ ";" ~ 'LineTerminator)) /* ImportDeclaration//1 */),
  'BindingElement := (('SingleNameBinding) /* BindingElement//0 */ | (('BindingPattern ~ (('WhiteSpace ~ 'Initializer)).?)) /* BindingElement//1 */),
  'ReturnStatement := (((" return " ~ ";")) /* ReturnStatement//0 */ | ((" return " ~ 'Expression ~ ";")) /* ReturnStatement//1 */),
  'Arguments := ((("(" ~ ")")) /* Arguments//0 */ | (("(" ~ 'ArgumentList ~ ")")) /* Arguments//1 */),
  'ForBinding := (('BindingIdentifier) /* ForBinding//0 */ | ('BindingPattern) /* ForBinding//1 */),
  'FromClause := (" from " ~ 'WhiteSpace ~ 'ModuleSpecifier),
  'TemplateInlay := ("${" ~ 'Expression ~ "}"),
  'FunctionStatementList := ('StatementList).?,
  'TemplateLiteral := (('NoSubstitutionTemplate) /* TemplateLiteral//0 */ | ('SubstitutionTemplate) /* TemplateLiteral//1 */),
  'RelationalExpression := (('ShiftExpression) /* RelationalExpression//0 */ | (('RelationalExpression ~ "<" ~ 'ShiftExpression)) /* RelationalExpression//1 */ | (('RelationalExpression ~ "<=" ~ 'ShiftExpression)) /* RelationalExpression//2 */ | (('RelationalExpression ~ ">" ~ 'ShiftExpression)) /* RelationalExpression//3 */ | (('RelationalExpression ~ ">=" ~ 'ShiftExpression)) /* RelationalExpression//4 */ | (('RelationalExpression ~ " in " ~ 'ShiftExpression)) /* RelationalExpression//5 */ | (('RelationalExpression ~ " instanceof " ~ 'ShiftExpression)) /* RelationalExpression//6 */),
  'TemplateCharacters := ('TemplateCharacter).rep(1),
  'NonemptyClassRanges := (('ClassAtom) /* NonemptyClassRanges//0 */ | (('ClassAtom ~ "-" ~ 'ClassAtom ~ 'ClassRanges)) /* NonemptyClassRanges//1 */ | (('ClassAtom ~ 'NonemptyClassRangesNoDash)) /* NonemptyClassRanges//2 */),
  'DecimalLiteral := ((("." ~ 'DecimalDigits ~ ('ExponentPart).?)) /* DecimalLiteral//0 */ | (('DecimalIntegerLiteral ~ "." ~ ('DecimalDigits).? ~ ('ExponentPart).?)) /* DecimalLiteral//1 */ | (('DecimalIntegerLiteral ~ ('ExponentPart).?)) /* DecimalLiteral//2 */),
  'VariableStatement := (" var " ~ 'WhiteSpace ~ 'VariableDeclarationList ~ ";"),
  'HexEscapeSequence := ("x" ~ 'HexDigit ~ 'HexDigit),
  'Script := 'ScriptBody,
  'PropertySetParameterList := 'FormalParameter,
  'ObjectBindingPattern := ((("{" ~ "}")) /* ObjectBindingPattern//0 */ | (("{" ~ 'BindingPropertyList ~ "," ~ "}")) /* ObjectBindingPattern//1 */ | (("{" ~ 'BindingPropertyList ~ "}")) /* ObjectBindingPattern//2 */),
  'StrictFormalParameters := 'FormalParameters,
  'PropertyDefinition := (('CoverInitializedName) /* PropertyDefinition//0 */ | ('IdentifierReference) /* PropertyDefinition//1 */ | ('MethodDefinition) /* PropertyDefinition//2 */ | (('PropertyName ~ ":" ~ 'AssignmentExpression)) /* PropertyDefinition//3 */),
  'DecimalDigit := (("0") /* DecimalDigit//0 */ | ("1") /* DecimalDigit//1 */ | ("2") /* DecimalDigit//2 */ | ("3") /* DecimalDigit//3 */ | ("4") /* DecimalDigit//4 */ | ("5") /* DecimalDigit//5 */ | ("6") /* DecimalDigit//6 */ | ("7") /* DecimalDigit//7 */ | ("8") /* DecimalDigit//8 */ | ("9") /* DecimalDigit//9 */),
  'VariableDeclarationList := ('VariableDeclaration ~ (("," ~ 'VariableDeclaration)).rep),
  'ArgumentList := (('AssignmentExpression) /* ArgumentList//0 */ | (('ArgumentList ~ "," ~ 'AssignmentExpression)) /* ArgumentList//1 */ | (('ArgumentList ~ "," ~ (('AssignmentExpression ~ 'WhiteSpace)).rep(1))) /* ArgumentList//2 */ | ((('AssignmentExpression ~ 'WhiteSpace)).rep(1)) /* ArgumentList//3 */),
  'Statement := (('BlockStatement) /* Statement//0 */ | ('BreakStatement) /* Statement//1 */ | ('BreakableStatement) /* Statement//2 */ | ('ContinueStatement) /* Statement//3 */ | ('DebuggerStatement) /* Statement//4 */ | ('EmptyStatement) /* Statement//5 */ | ('ExpressionStatement) /* Statement//6 */ | ('IfStatement) /* Statement//7 */ | ('LabelledStatement) /* Statement//8 */ | ('ReturnStatement) /* Statement//9 */ | ('ThrowStatement) /* Statement//10 */ | ('TryStatement) /* Statement//11 */ | ('VariableStatement) /* Statement//12 */ | ('WithStatement) /* Statement//13 */),
  'NullLiteral := " null ",
  'LabelledItem := (('FunctionDeclaration) /* LabelledItem//0 */ | ('Statement) /* LabelledItem//1 */),
  'Declaration := (('ClassDeclaration) /* Declaration//0 */ | ('HoistableDeclaration) /* Declaration//1 */ | ('LexicalDeclaration) /* Declaration//2 */),
  'SuperCall := ("super" ~ 'Arguments),
  'DebuggerStatement := ("debugger" ~ ";"),
  'VariableDeclaration := ((('BindingIdentifier ~ (('WhiteSpace ~ 'Initializer)).?)) /* VariableDeclaration//0 */ | (('BindingPattern ~ 'WhiteSpace ~ 'Initializer)) /* VariableDeclaration//1 */),
  'LexicalDeclaration := ('LetOrConst ~ 'WhiteSpace ~ 'BindingList ~ ";"),
  'LetOrConst := ((" const ") /* LetOrConst//0 */ | (" let ") /* LetOrConst//1 */),
  'ModuleDummy2 := 'ModuleDummy1,
  'PostfixExpression := (('LeftHandSideExpression) /* PostfixExpression//0 */ | (('LeftHandSideExpression ~ "++")) /* PostfixExpression//1 */ | (('LeftHandSideExpression ~ "--")) /* PostfixExpression//2 */),
  'ClassAtomNoDash := ((("\\" ~ 'ClassEscape)) /* ClassAtomNoDash//0 */ | ("[^\\\\]-]".regex) /* ClassAtomNoDash//1 */),
  'ClassTail := (('ClassHeritage).? ~ "{" ~ ('LineTerminator).? ~ ('ClassBody).? ~ "}"),
  'IdentityEscape := (("/") /* IdentityEscape//0 */ | ('SyntaxCharacter) /* IdentityEscape//1 */ | (".".regex) /* IdentityEscape//2 */),
  'ClassHeritage := (" extends" ~ 'WhiteSpace ~ 'LeftHandSideExpression),
  'ImportsList := ('ImportSpecifier ~ (("," ~ 'ImportSpecifier)).rep),
  'TryStatement := (((" try " ~ 'Block ~ 'Catch ~ 'Finally)) /* TryStatement//0 */ | ((" try " ~ 'Block ~ 'Catch)) /* TryStatement//1 */ | ((" try " ~ 'Block ~ 'Finally)) /* TryStatement//2 */),
  'BindingRestElement := (('BindingIdentifier ~ 'WhiteSpace)).rep(1),
  'ForDeclaration := ('LetOrConst ~ 'WhiteSpace ~ 'ForBinding),
  'LabelledStatement := ('LabelIdentifier ~ ":" ~ 'LabelledItem),
  'ClassRanges := (("") /* ClassRanges//0 */ | ('NonemptyClassRanges) /* ClassRanges//1 */),
  'MethodDefinition := (('GeneratorMethod) /* MethodDefinition//0 */ | ((" constructor " ~ "(" ~ 'StrictFormalParameters ~ ") {" ~ 'FunctionBody ~ "}")) /* MethodDefinition//1 */ | (("get" ~ 'PropertyName ~ "(" ~ ")" ~ "{" ~ 'FunctionBody ~ "}")) /* MethodDefinition//2 */ | (("set" ~ 'PropertyName ~ "(" ~ 'PropertySetParameterList ~ ") {" ~ 'FunctionBody ~ "}")) /* MethodDefinition//3 */ | (((("async" ~ 'WhiteSpace)).? ~ 'PropertyName ~ "(" ~ 'StrictFormalParameters ~ ") {" ~ 'FunctionBody ~ "}")) /* MethodDefinition//4 */),
  'EscapeSequence := (("0") /* EscapeSequence//0 */ | ('CharacterEscapeSequence) /* EscapeSequence//1 */ | ('HexEscapeSequence) /* EscapeSequence//2 */ | ('UnicodeEscapeSequence) /* EscapeSequence//3 */),
  'LogicalANDExpression := (('BitwiseORExpression) /* LogicalANDExpression//0 */ | (('LogicalANDExpression ~ "&&" ~ 'BitwiseORExpression)) /* LogicalANDExpression//1 */),
  'HexDigit := "[0-9a-fA-F]".regex,
  'CharacterClass := ((("[" ~ "^" ~ 'ClassRanges ~ "]")) /* CharacterClass//0 */ | (("[" ~ 'ClassRanges ~ "]")) /* CharacterClass//1 */),
  'BinaryDigit := (("0") /* BinaryDigit//0 */ | ("1") /* BinaryDigit//1 */),
  'MultiplicativeOperator := (("%") /* MultiplicativeOperator//0 */ | ("*") /* MultiplicativeOperator//1 */ | ("/") /* MultiplicativeOperator//2 */),
  'OctalIntegerLiteral := ((("0O" ~ 'OctalDigits)) /* OctalIntegerLiteral//0 */ | (("0o" ~ 'OctalDigits)) /* OctalIntegerLiteral//1 */),
  'LineTerminator := 'LF,
  'ModuleBody := 'ModuleItemList,
  'FunctionExpression := (" function " ~ ('BindingIdentifier).? ~ "(" ~ 'FormalParameters ~ ")" ~ "{" ~ ('LineTerminator).? ~ 'FunctionBody ~ "}"),
  'GeneratorBody := 'FunctionBody,
  'Identifier := 'IdentifierName,
  'ControlEscape := (("f") /* ControlEscape//0 */ | ("n") /* ControlEscape//1 */ | ("r") /* ControlEscape//2 */ | ("t") /* ControlEscape//3 */ | ("v") /* ControlEscape//4 */),
  'BinaryDigits := (('BinaryDigit) /* BinaryDigits//0 */ | (('BinaryDigits ~ 'BinaryDigit)) /* BinaryDigits//1 */),
  'Elision := ((",") /* Elision//0 */ | (('Elision ~ ",")) /* Elision//1 */),
  'LexicalBinding := ((('BindingIdentifier ~ 'WhiteSpace ~ 'Initializer)) /* LexicalBinding//0 */ | (('BindingPattern ~ 'WhiteSpace ~ 'Initializer)) /* LexicalBinding//1 */),
  'ExportSpecifier := (('IdentifierName) /* ExportSpecifier//0 */ | (('IdentifierName ~ " as " ~ 'IdentifierName)) /* ExportSpecifier//1 */),
  'LF := "\n",
  'ThrowStatement := (" throw " ~ 'WhiteSpace ~ 'Expression ~ ";"),
  'ClassExpression := (" class " ~ 'WhiteSpace ~ ('BindingIdentifier).? ~ 'ClassTail),
  'HoistableDeclaration := (('FunctionDeclaration) /* HoistableDeclaration//0 */ | ('GeneratorDeclaration) /* HoistableDeclaration//1 */),
  'WithStatement := (" with " ~ "(" ~ 'Expression ~ ")" ~ 'Statement),
  'CharacterEscapeSequence := (('NonEscapeCharacter) /* CharacterEscapeSequence//0 */ | ('SingleEscapeCharacter) /* CharacterEscapeSequence//1 */),
  'UnicodeIDContinue := ".".regex,
  'IdentifierName := 'IdentifierStart ~ 'IdentifierPart.rep,
  'IdentifierStart := (('UnicodeLetter) /* IdentifierStart//0 */ | (("\\" ~ 'UnicodeEscapeSequence)) /* IdentifierStart//1 */ | ("[$_]".regex) /* IdentifierStart//2 */),
  'PropertyDefinitionList := ((('PropertyDefinition ~ 'WhiteSpace)) /* PropertyDefinitionList//0 */ | (('PropertyDefinitionList ~ "," ~ 'PropertyDefinition)) /* PropertyDefinitionList//1 */),
  'ModuleItem := (('ExportDeclaration) /* ModuleItem//0 */ | ('ImportDeclaration) /* ModuleItem//1 */ | ('StatementListItem) /* ModuleItem//2 */),
  'DecimalEscape := 'DecimalIntegerLiteral,
  'StatementListItem := (('Declaration) /* StatementListItem//0 */ | ('Statement) /* StatementListItem//1 */),
  'ClassEscape := (("-") /* ClassEscape//0 */ | ("b") /* ClassEscape//1 */ | ('CharacterClassEscape) /* ClassEscape//2 */ | ('CharacterEscape) /* ClassEscape//3 */ | ('DecimalEscape) /* ClassEscape//4 */),
  'SpreadElement := (('AssignmentExpression ~ 'WhiteSpace)).rep(1),
  'ConciseBody := (('AssignmentExpression) /* ConciseBody//0 */ | (("{" ~ 'FunctionBody ~ "}")) /* ConciseBody//1 */),
  'LiteralPropertyName := (('IdentifierName) /* LiteralPropertyName//0 */ | ('NumericLiteral) /* LiteralPropertyName//1 */ | ('StringLiteral) /* LiteralPropertyName//2 */),
  'ExponentPart := (('ExponentIndicator) /* ExponentPart//0 */ | ('SignedInteger) /* ExponentPart//1 */),
  'FormalsList := (('FormalParameter) /* FormalsList//0 */ | (('FormalsList ~ "," ~ 'WhiteSpace ~ 'FormalParameter)) /* FormalsList//1 */),
  'ExponentIndicator := (("E") /* ExponentIndicator//0 */ | ("e") /* ExponentIndicator//1 */),
  'TrailSurrogate := 'Hex4Digits,
  'SuperProperty := ((("super" ~ "." ~ 'IdentifierName)) /* SuperProperty//0 */ | (("super" ~ "[" ~ 'Expression ~ "]")) /* SuperProperty//1 */),
  'LeftHandSideExpression := (('CallExpression) /* LeftHandSideExpression//0 */ | ('NewExpression) /* LeftHandSideExpression//1 */),
  'BreakStatement := (("break;") /* BreakStatement//0 */ | (("break " ~ 'LabelIdentifier ~ ";")) /* BreakStatement//1 */),
  'FunctionBody := 'FunctionStatementList,
  'ClassElementList := ((('ClassElement ~ ('LineTerminator).?)) /* ClassElementList//0 */ | (('ClassElementList ~ 'ClassElement ~ ('LineTerminator).?)) /* ClassElementList//1 */),
  'Disjunction := (('Alternative) /* Disjunction//0 */ | (('Alternative ~ "|" ~ 'Disjunction)) /* Disjunction//1 */),
  'AssignmentOperator := (("%=") /* AssignmentOperator//0 */ | ("*=") /* AssignmentOperator//1 */ | ("+=") /* AssignmentOperator//2 */ | ("-=") /* AssignmentOperator//3 */ | ("/=") /* AssignmentOperator//4 */ | ("=") /* AssignmentOperator//5 */ | (">>=") /* AssignmentOperator//6 */ | ("^=") /* AssignmentOperator//7 */ | ("|=") /* AssignmentOperator//8 */),
  'LabelIdentifier := 'Identifier,
  'CharacterEscape := (('ControlEscape) /* CharacterEscape//0 */ | ('HexEscapeSequence) /* CharacterEscape//1 */ | ('IdentityEscape) /* CharacterEscape//2 */ | ('RegExpUnicodeEscapeSequence) /* CharacterEscape//3 */ | (("c" ~ 'ControlLetter)) /* CharacterEscape//4 */),
  'BindingList := (('LexicalBinding) /* BindingList//0 */ | (('BindingList ~ "," ~ 'LexicalBinding)) /* BindingList//1 */),
  'Catch := (" catch " ~ "(" ~ 'CatchParameter ~ ")" ~ 'Block),
  'Alternative := (("") /* Alternative//0 */ | (('Alternative ~ 'Term)) /* Alternative//1 */),
  'NoSubstitutionTemplate := ("`" ~ ('TemplateCharacters).? ~ "`"),
  'NumericLiteral := (('BinaryIntegerLiteral) /* NumericLiteral//0 */ | ('DecimalLiteral) /* NumericLiteral//1 */ | ('HexIntegerLiteral) /* NumericLiteral//2 */ | ('OctalIntegerLiteral) /* NumericLiteral//3 */),
  'Block := ("{" ~ ('LineTerminator).? ~ ('StatementList).? ~ "}" ~ 'LineTerminator),
  'FunctionDeclaration := (((" function " ~ "(" ~ 'FormalParameters ~ ")" ~ "{" ~ 'FunctionBody ~ "}")) /* FunctionDeclaration//0 */ | ((" function " ~ 'WhiteSpace ~ 'BindingIdentifier ~ "(" ~ 'FormalParameters ~ ")" ~ "{" ~ 'FunctionBody ~ "}")) /* FunctionDeclaration//1 */),
  'Assertion := (("$") /* Assertion//0 */ | ("\\\\B") /* Assertion//1 */ | ("\\\\b") /* Assertion//2 */ | ("^") /* Assertion//3 */ | (("(" ~ "?" ~ "!" ~ 'Disjunction ~ ")")) /* Assertion//4 */ | (("(" ~ "?" ~ "=" ~ 'Disjunction ~ ")")) /* Assertion//5 */),
  'CharacterClassEscape := (("D") /* CharacterClassEscape//0 */ | ("S") /* CharacterClassEscape//1 */ | ("W") /* CharacterClassEscape//2 */ | ("d") /* CharacterClassEscape//3 */ | ("s") /* CharacterClassEscape//4 */ | ("w") /* CharacterClassEscape//5 */),
  'IdentifierPart := (("$") /* IdentifierPart//0 */ |
    ("_") /* IdentifierPart//1 */ |
    ('UnicodeIDContinue) /* IdentifierPart//2 */ |
    (("\\" ~ 'UnicodeEscapeSequence)) /* IdentifierPart//3 */),
  'HexIntegerLiteral := ((("0X" ~ 'HexDigits)) /* HexIntegerLiteral//0 */ | (("0x" ~ 'HexDigits)) /* HexIntegerLiteral//1 */),
  'Initializer := ("=" ~ 'WhiteSpace ~ 'AssignmentExpression),
  'SingleStringCharacters := ('SingleStringCharacter).rep(1),
  'ControlLetter := (("A") /* ControlLetter//0 */ | ("B") /* ControlLetter//1 */ | ("C") /* ControlLetter//2 */ | ("D") /* ControlLetter//3 */ | ("E") /* ControlLetter//4 */ | ("F") /* ControlLetter//5 */ | ("G") /* ControlLetter//6 */ | ("H") /* ControlLetter//7 */ | ("I") /* ControlLetter//8 */ | ("J") /* ControlLetter//9 */ | ("K") /* ControlLetter//10 */ | ("L") /* ControlLetter//11 */ | ("M") /* ControlLetter//12 */ | ("N") /* ControlLetter//13 */ | ("O") /* ControlLetter//14 */ | ("P") /* ControlLetter//15 */ | ("Q") /* ControlLetter//16 */ | ("R") /* ControlLetter//17 */ | ("S") /* ControlLetter//18 */ | ("T") /* ControlLetter//19 */ | ("U") /* ControlLetter//20 */ | ("V") /* ControlLetter//21 */ | ("W") /* ControlLetter//22 */ | ("X") /* ControlLetter//23 */ | ("Y") /* ControlLetter//24 */ | ("Z") /* ControlLetter//25 */ | ("a") /* ControlLetter//26 */ | ("b") /* ControlLetter//27 */ | ("c") /* ControlLetter//28 */ | ("d") /* ControlLetter//29 */ | ("e") /* ControlLetter//30 */ | ("f") /* ControlLetter//31 */ | ("g") /* ControlLetter//32 */ | ("h") /* ControlLetter//33 */ | ("i") /* ControlLetter//34 */ | ("j") /* ControlLetter//35 */ | ("k") /* ControlLetter//36 */ | ("l") /* ControlLetter//37 */ | ("m") /* ControlLetter//38 */ | ("n") /* ControlLetter//39 */ | ("o") /* ControlLetter//40 */ | ("p") /* ControlLetter//41 */ | ("q") /* ControlLetter//42 */ | ("r") /* ControlLetter//43 */ | ("s") /* ControlLetter//44 */ | ("t") /* ControlLetter//45 */ | ("u") /* ControlLetter//46 */ | ("v") /* ControlLetter//47 */ | ("w") /* ControlLetter//48 */ | ("x") /* ControlLetter//49 */ | ("y") /* ControlLetter//50 */ | ("z") /* ControlLetter//51 */),
  'BooleanLiteral := (("false") /* BooleanLiteral//0 */ | ("true") /* BooleanLiteral//1 */),
  'SingleEscapeCharacter := (("'") /* SingleEscapeCharacter//0 */ | ("\"") /* SingleEscapeCharacter//1 */ | ("\\") /* SingleEscapeCharacter//2 */ | ("b") /* SingleEscapeCharacter//3 */ | ("f") /* SingleEscapeCharacter//4 */ | ("n") /* SingleEscapeCharacter//5 */ | ("r") /* SingleEscapeCharacter//6 */ | ("t") /* SingleEscapeCharacter//7 */ | ("v") /* SingleEscapeCharacter//8 */),
  'BindingIdentifier := 'Identifier,
  'FormalParameter := 'BindingElement,
  'RegularExpressionFlags := (("") /* RegularExpressionFlags//0 */ | (('RegularExpressionFlags ~ 'IdentifierPart)) /* RegularExpressionFlags//1 */),
  'Finally := (" finally " ~ 'Block),
  'Module := 'ModuleBody,
  'FormalParameterList := (('FormalsList) /* FormalParameterList//0 */ | ('FunctionRestParameter) /* FormalParameterList//1 */ | (('FormalsList ~ "," ~ 'WhiteSpace ~ 'FunctionRestParameter)) /* FormalParameterList//2 */),
  'RegularExpressionLiteral := ("/" ~ 'Pattern ~ "/" ~ 'RegularExpressionFlags),
  'ElementList := 'Element ~ ("," ~ 'Element).rep,
  'SwitchStatement := (" switch " ~ "(" ~ 'Expression ~ ")" ~ 'CaseBlock),
  'ImportedBinding := 'BindingIdentifier,
  'LineContinuation := (("\\") /* LineContinuation//0 */ | ('LineTerminatorSequence) /* LineContinuation//1 */),
  'ExpressionStatement := ('Expression ~ ";"),
  'ExportDeclaration := (((" export " ~ 'WhiteSpace ~ "*" ~ 'WhiteSpace ~ 'FromClause ~ ";" ~ 'LineTerminator)) /* ExportDeclaration//0 */ | ((" export" ~ 'WhiteSpace ~ "default" ~ 'WhiteSpace ~ 'AssignmentExpression ~ ";" ~ 'LineTerminator)) /* ExportDeclaration//1 */ | ((" export" ~ 'WhiteSpace ~ "default" ~ 'WhiteSpace ~ 'ClassDeclaration ~ 'LineTerminator)) /* ExportDeclaration//2 */ | ((" export" ~ 'WhiteSpace ~ "default" ~ 'WhiteSpace ~ 'HoistableDeclaration ~ 'LineTerminator)) /* ExportDeclaration//3 */ | ((" export" ~ 'WhiteSpace ~ 'Declaration ~ 'LineTerminator)) /* ExportDeclaration//4 */ | ((" export" ~ 'WhiteSpace ~ 'ExportClause ~ ";" ~ 'LineTerminator)) /* ExportDeclaration//5 */ | ((" export" ~ 'WhiteSpace ~ 'ExportClause ~ 'FromClause ~ ";" ~ 'LineTerminator)) /* ExportDeclaration//6 */ | ((" export" ~ 'WhiteSpace ~ 'VariableStatement ~ 'LineTerminator)) /* ExportDeclaration//7 */),
  'AdditiveExpression := (('MultiplicativeExpression) /* AdditiveExpression//0 */ | (('AdditiveExpression ~ "+" ~ 'MultiplicativeExpression)) /* AdditiveExpression//1 */ | (('AdditiveExpression ~ "-" ~ 'MultiplicativeExpression)) /* AdditiveExpression//2 */),
  'UnicodeLetter
    := "[\u0041-\u005A]".regex
    | "[\u0061-\u007A]".regex
    | "\u00AA"
    | "\u00B5"
    | "\u00BA"
    | "[\u00C0-\u00D6]".regex
    | "[\u00D8-\u00F6]".regex
    | "[\u00F8-\u021F]".regex
    | "[\u0222-\u0233]".regex
    | "[\u0250-\u02AD]".regex
    | "[\u02B0-\u02B8]".regex
    | "[\u02BB-\u02C1]".regex
    | "[\u02D0-\u02D1]".regex
    | "[\u02E0-\u02E4]".regex
    | "\u02EE"
    | "\u037A"
    | "\u0386"
    | "[\u0388-\u038A]".regex
    | "\u038C"
    | "[\u038E-\u03A1]".regex
    | "[\u03A3-\u03CE]".regex
    | "[\u03D0-\u03D7]".regex
    | "[\u03DA-\u03F3]".regex
    | "[\u0400-\u0481]".regex
    | "[\u048C-\u04C4]".regex
    | "[\u04C7-\u04C8]".regex
    | "[\u04CB-\u04CC]".regex
    | "[\u04D0-\u04F5]".regex
    | "[\u04F8-\u04F9]".regex
    | "[\u0531-\u0556]".regex
    | "\u0559"
    | "[\u0561-\u0587]".regex
    | "[\u05D0-\u05EA]".regex
    | "[\u05F0-\u05F2]".regex
    | "[\u0621-\u063A]".regex
    | "[\u0640-\u064A]".regex
    | "[\u0671-\u06D3]".regex
    | "\u06D5"
    | "[\u06E5-\u06E6]".regex
    | "[\u06FA-\u06FC]".regex
    | "\u0710"
    | "[\u0712-\u072C]".regex
    | "[\u0780-\u07A5]".regex
    | "[\u0905-\u0939]".regex
    | "\u093D"
    | "\u0950"
    | "[\u0958-\u0961]".regex
    | "[\u0985-\u098C]".regex
    | "[\u098F-\u0990]".regex
    | "[\u0993-\u09A8]".regex
    | "[\u09AA-\u09B0]".regex
    | "\u09B2"
    | "[\u09B6-\u09B9]".regex
    | "[\u09DC-\u09DD]".regex
    | "[\u09DF-\u09E1]".regex
    | "[\u09F0-\u09F1]".regex
    | "[\u0A05-\u0A0A]".regex
    | "[\u0A0F-\u0A10]".regex
    | "[\u0A13-\u0A28]".regex
    | "[\u0A2A-\u0A30]".regex
    | "[\u0A32-\u0A33]".regex
    | "[\u0A35-\u0A36]".regex
    | "[\u0A38-\u0A39]".regex
    | "[\u0A59-\u0A5C]".regex
    | "\u0A5E"
    | "[\u0A72-\u0A74]".regex
    | "[\u0A85-\u0A8B]".regex
    | "\u0A8D"
    | "[\u0A8F-\u0A91]".regex
    | "[\u0A93-\u0AA8]".regex
    | "[\u0AAA-\u0AB0]".regex
    | "[\u0AB2-\u0AB3]".regex
    | "[\u0AB5-\u0AB9]".regex
    | "\u0ABD"
    | "\u0AD0"
    | "\u0AE0"
    | "[\u0B05-\u0B0C]".regex
    | "[\u0B0F-\u0B10]".regex
    | "[\u0B13-\u0B28]".regex
    | "[\u0B2A-\u0B30]".regex
    | "[\u0B32-\u0B33]".regex
    | "[\u0B36-\u0B39]".regex
    | "\u0B3D"
    | "[\u0B5C-\u0B5D]".regex
    | "[\u0B5F-\u0B61]".regex
    | "[\u0B85-\u0B8A]".regex
    | "[\u0B8E-\u0B90]".regex
    | "[\u0B92-\u0B95]".regex
    | "[\u0B99-\u0B9A]".regex
    | "\u0B9C"
    | "[\u0B9E-\u0B9F]".regex
    | "[\u0BA3-\u0BA4]".regex
    | "[\u0BA8-\u0BAA]".regex
    | "[\u0BAE-\u0BB5]".regex
    | "[\u0BB7-\u0BB9]".regex
    | "[\u0C05-\u0C0C]".regex
    | "[\u0C0E-\u0C10]".regex
    | "[\u0C12-\u0C28]".regex
    | "[\u0C2A-\u0C33]".regex
    | "[\u0C35-\u0C39]".regex
    | "[\u0C60-\u0C61]".regex
    | "[\u0C85-\u0C8C]".regex
    | "[\u0C8E-\u0C90]".regex
    | "[\u0C92-\u0CA8]".regex
    | "[\u0CAA-\u0CB3]".regex
    | "[\u0CB5-\u0CB9]".regex
    | "\u0CDE"
    | "[\u0CE0-\u0CE1]".regex
    | "[\u0D05-\u0D0C]".regex
    | "[\u0D0E-\u0D10]".regex
    | "[\u0D12-\u0D28]".regex
    | "[\u0D2A-\u0D39]".regex
    | "[\u0D60-\u0D61]".regex
    | "[\u0D85-\u0D96]".regex
    | "[\u0D9A-\u0DB1]".regex
    | "[\u0DB3-\u0DBB]".regex
    | "\u0DBD"
    | "[\u0DC0-\u0DC6]".regex
    | "[\u0E01-\u0E30]".regex
    | "[\u0E32-\u0E33]".regex
    | "[\u0E40-\u0E46]".regex
    | "[\u0E81-\u0E82]".regex
    | "\u0E84"
    | "[\u0E87-\u0E88]".regex
    | "\u0E8A"
    | "\u0E8D"
    | "[\u0E94-\u0E97]".regex
    | "[\u0E99-\u0E9F]".regex
    | "[\u0EA1-\u0EA3]".regex
    | "\u0EA5"
    | "\u0EA7"
    | "[\u0EAA-\u0EAB]".regex
    | "[\u0EAD-\u0EB0]".regex
    | "[\u0EB2-\u0EB3]".regex
    | "[\u0EBD-\u0EC4]".regex
    | "\u0EC6"
    | "[\u0EDC-\u0EDD]".regex
    | "\u0F00"
    | "[\u0F40-\u0F6A]".regex
    | "[\u0F88-\u0F8B]".regex
    | "[\u1000-\u1021]".regex
    | "[\u1023-\u1027]".regex
    | "[\u1029-\u102A]".regex
    | "[\u1050-\u1055]".regex
    | "[\u10A0-\u10C5]".regex
    | "[\u10D0-\u10F6]".regex
    | "[\u1100-\u1159]".regex
    | "[\u115F-\u11A2]".regex
    | "[\u11A8-\u11F9]".regex
    | "[\u1200-\u1206]".regex
    | "[\u1208-\u1246]".regex
    | "\u1248"
    | "[\u124A-\u124D]".regex
    | "[\u1250-\u1256]".regex
    | "\u1258"
    | "[\u125A-\u125D]".regex
    | "[\u1260-\u1286]".regex
    | "\u1288"
    | "[\u128A-\u128D]".regex
    | "[\u1290-\u12AE]".regex
    | "\u12B0"
    | "[\u12B2-\u12B5]".regex
    | "[\u12B8-\u12BE]".regex
    | "\u12C0"
    | "[\u12C2-\u12C5]".regex
    | "[\u12C8-\u12CE]".regex
    | "[\u12D0-\u12D6]".regex
    | "[\u12D8-\u12EE]".regex
    | "[\u12F0-\u130E]".regex
    | "\u1310"
    | "[\u1312-\u1315]".regex
    | "[\u1318-\u131E]".regex
    | "[\u1320-\u1346]".regex
    | "[\u1348-\u135A]".regex
    | "[\u13A0-\u13B0]".regex
    | "[\u13B1-\u13F4]".regex
    | "[\u1401-\u1676]".regex
    | "[\u1681-\u169A]".regex
    | "[\u16A0-\u16EA]".regex
    | "[\u1780-\u17B3]".regex
    | "[\u1820-\u1877]".regex
    | "[\u1880-\u18A8]".regex
    | "[\u1E00-\u1E9B]".regex
    | "[\u1EA0-\u1EE0]".regex
    | "[\u1EE1-\u1EF9]".regex
    | "[\u1F00-\u1F15]".regex
    | "[\u1F18-\u1F1D]".regex
    | "[\u1F20-\u1F39]".regex
    | "[\u1F3A-\u1F45]".regex
    | "[\u1F48-\u1F4D]".regex
    | "[\u1F50-\u1F57]".regex
    | "\u1F59"
    | "\u1F5B"
    | "\u1F5D"
    | "[\u1F5F-\u1F7D]".regex
    | "[\u1F80-\u1FB4]".regex
    | "[\u1FB6-\u1FBC]".regex
    | "\u1FBE"
    | "[\u1FC2-\u1FC4]".regex
    | "[\u1FC6-\u1FCC]".regex
    | "[\u1FD0-\u1FD3]".regex
    | "[\u1FD6-\u1FDB]".regex
    | "[\u1FE0-\u1FEC]".regex
    | "[\u1FF2-\u1FF4]".regex
    | "[\u1FF6-\u1FFC]".regex
    | "\u207F"
    | "\u2102"
    | "\u2107"
    | "[\u210A-\u2113]".regex
    | "\u2115"
    | "[\u2119-\u211D]".regex
    | "\u2124"
    | "\u2126"
    | "\u2128"
    | "[\u212A-\u212D]".regex
    | "[\u212F-\u2131]".regex
    | "[\u2133-\u2139]".regex
    | "[\u2160-\u2183]".regex
    | "[\u3005-\u3007]".regex
    | "[\u3021-\u3029]".regex
    | "[\u3031-\u3035]".regex
    | "[\u3038-\u303A]".regex
    | "[\u3041-\u3094]".regex
    | "[\u309D-\u309E]".regex
    | "[\u30A1-\u30FA]".regex
    | "[\u30FC-\u30FE]".regex
    | "[\u3105-\u312C]".regex
    | "[\u3131-\u318E]".regex
    | "[\u31A0-\u31B7]".regex
    | "\u3400"
    | "\u4DB5"
    | "\u4E00"
    | "\u9FA5"
    | "[\uA000-\uA48C]".regex
    | "\uAC00"
    | "\uD7A3"
    | "[\uF900-\uFA2D]".regex
    | "[\uFB00-\uFB06]".regex
    | "[\uFB13-\uFB17]".regex
    | "\uFB1D"
    | "[\uFB1F-\uFB28]".regex
    | "[\uFB2A-\uFB36]".regex
    | "[\uFB38-\uFB3C]".regex
    | "\uFB3E"
    | "[\uFB40-\uFB41]".regex
    | "[\uFB43-\uFB44]".regex
    | "[\uFB46-\uFBB1]".regex
    | "[\uFBD3-\uFD3D]".regex
    | "[\uFD50-\uFD8F]".regex
    | "[\uFD92-\uFDC7]".regex
    | "[\uFDF0-\uFDFB]".regex
    | "[\uFE70-\uFE72]".regex
    | "\uFE74"
    | "[\uFE76-\uFEFC]".regex
    | "[\uFF21-\uFF3A]".regex
    | "[\uFF41-\uFF5A]".regex
    | "[\uFF66-\uFFBE]".regex
    | "[\uFFC2-\uFFC7]".regex
    | "[\uFFCA-\uFFCF]".regex
    | "[\uFFD2-\uFFD7]".regex
    | "[\uFFDA-\uFFDC]".regex
)
