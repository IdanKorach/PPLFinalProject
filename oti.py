from strings_with_arrows import *
import string

#################################
# CONSTANTS
#################################

DIGITS = '0123456789'

CHARS = string.ascii_letters

#################################
# ERRORS
#################################

class Error:
    def __init__(self, pos_start, pos_end, error_name, details):
        self.pos_start = pos_start
        self.pos_end = pos_end
        self.error_name = error_name
        self.details = details
    
    def as_string(self):
        result = f'{self.error_name}: {self.details}'
        result += f'File {self.pos_start.fn}, ln {self.pos_end.ln + 1}'
        result += '\n\n' + string_with_arrows(self.pos_start.ftxt, self.pos_start, self.pos_end)
        return result

class IllegalCharError(Error):
    def __init__(self, pos_start, pos_end, details):
        super().__init__(pos_start, pos_end, 'Illegal character\n', details)

class InvalidSyntaxError(Error):
    def __init__(self, pos_start, pos_end, details):
        super().__init__(pos_start, pos_end, 'Illegal syntax\n', details)

class RTError(Error):
    def __init__(self, pos_start, pos_end, details, context):
        super().__init__(pos_start, pos_end, 'Runtime Error\n', details)
        self.context = context
    
    def as_string(self):
        result = self.generate_traceback()
        result += f'{self.error_name}: {self.details}'
        result += '\n\n' + string_with_arrows(self.pos_start.ftxt, self.pos_start, self.pos_end)
        return result

    def generate_traceback(self):
        result = ''
        pos = self.pos_start
        ctx = self.context

        while ctx:
            result = f'  File {pos.fn}, line {str(pos.ln + 1)}, in {ctx.display_name}\n' + result
            pos = ctx.parent_entry_pos
            ctx = ctx.parent
        
        return 'Traceback (most recent call last):\n' + result

#################################
# POSITION
#################################

class Position:
    def __init__(self, idx, ln, col, fn, ftxt): # fn = file name, ftxt = file text
        self.idx = idx
        self.ln = ln
        self.col = col
        self.fn = fn
        self.ftxt = ftxt
    
    def advance(self, current_char=None):
        self.idx += 1
        self.col += 1

        if current_char == '\n':
            self.ln += 1
            self.col = 0

        return self

    def copy(self):
        return Position(self.idx, self.ln, self.col, self.fn, self.ftxt)

#################################
# TOKENS
#################################

TT_INT      = 'INT'
TT_FLOAT    = 'FLOAT'
TT_PLUS     = 'PLUS'
TT_MINUS    = 'MINUS'
TT_MUL      = 'MUL'
TT_DIV      = 'DIV'
TT_POWER    = 'POW'
TT_LPAREN   = 'LPAREN'
TT_RPAREN   = 'RPAREN'
TT_EOF      = 'EOF'
TT_MIN      = 'MIN'
TT_MAX      = 'MAX'
TT_COMMA    = 'COMMA'
TT_ASSIGN   = 'ASSIGN'
TT_EQUAL    = 'EQUAL'
TT_NEQUAL   = 'NEQUAL'
TT_AND      = 'AND'
TT_OR       = 'OR'
TT_NOT      = 'NOT'
TT_GT       = 'GTHEN'
TT_LT       = 'LTHEN'
TT_GTE      = 'GTEQUAL'
TT_LTE      = 'LTEQUAL'
TT_VAR      = 'VAR'
TT_SNGLQTE  = 'SNGLQUOTE'
TT_DBLQTE   = 'DBLQUOTE'
TT_NEWLINE  = 'NEWLINE'


SAVED_WORDS = ['Min', 'Max']

class Token:
    def __init__(self, type_, value=None, pos_start=0, pos_end=0):
        self.type = type_
        self.value = value

        if pos_start:
            self.pos_start = pos_start.copy()
            self.pos_end = pos_start.copy()
            self.pos_end.advance()

        if pos_end:
            self.pos_end = pos_end

    def __repr__(self):
        if self.value: 
            return f'{self.type}:{self.value}'
        return f'{self.type}'
    
#################################
# LEXER
#################################
    
class Lexer:
    def __init__(self, fn, text):
        self.fn = fn
        self.text = text
        self.pos = Position(-1, 0, -1, fn, text)
        self.current_char = None
        self.advance()

    def advance(self, steps=1):
        for _ in range(steps):
            self.pos.advance(self.current_char)
            self.current_char = self.text[self.pos.idx] if self.pos.idx < len(self.text) else None

    def peek(self, steps=1):
        peek_pos = self.pos.idx + steps
        if peek_pos >= len(self.text):
            return None
        return self.text[peek_pos]

    def make_tokens(self):
        tokens = []

        while self.current_char != None:

            if self.current_char in ' \t':
                self.advance()

            elif self.current_char in DIGITS:
                tokens.append(self.make_number())

            elif self.current_char == '+':
                tokens.append(Token(TT_PLUS, pos_start=self.pos))
                self.advance()

            elif self.current_char == '-':
                tokens.append(Token(TT_MINUS, pos_start=self.pos))
                self.advance()

            elif self.current_char == '*':
                tokens.append(Token(TT_MUL, pos_start=self.pos ))
                self.advance()

            elif self.current_char == '/':
                tokens.append(Token(TT_DIV, pos_start=self.pos))
                self.advance()

            elif self.current_char == '(':
                tokens.append(Token(TT_LPAREN, pos_start=self.pos))
                self.advance()

            elif self.current_char == ')':
                tokens.append(Token(TT_RPAREN, pos_start=self.pos))  
                self.advance()  

            elif self.current_char == '^':
                tokens.append(Token(TT_POWER, pos_start=self.pos))
                self.advance()

            elif self.current_char == ',':
                tokens.append(Token(TT_COMMA, pos_start=self.pos))
                self.advance()

            elif self.current_char == 'M' and self.peek() == 'i':
                if self.peek(2) == 'n':
                    tokens.append(Token(TT_MIN, pos_start=self.pos))
                    self.advance(3)  # Advance 3 times to skip 'Min'
                else:
                    self.advance()

            elif self.current_char == 'M' and self.peek() == 'a':
                if self.peek(2) == 'x':
                    tokens.append(Token(TT_MAX, pos_start=self.pos))
                    self.advance(3)  # Advance 3 times to skip 'Min'
                else:
                    self.advance()

            elif self.current_char in CHARS:    
                tokens.append(self.make_variable())

            elif self.current_char == '=': 
                if self.peek() == '=':
                    tokens.append(Token(TT_EQUAL, pos_start=self.pos))
                    self.advance(2)
                else:
                    tokens.append(Token(TT_ASSIGN, pos_start=self.pos))
                    self.advance()

            elif self.current_char == '!': 
                if self.peek() == '=':
                    tokens.append(Token(TT_NEQUAL, pos_start=self.pos))
                    self.advance(2)
                else:
                    tokens.append(Token(TT_NOT, pos_start=self.pos))
                    self.advance()

            elif self.current_char == '>':
                if self.peek() == '=':
                    tokens.append(Token(TT_GTE, pos_start=self.pos))
                    self.advance(2)
                else:
                    tokens.append(Token(TT_GT, pos_start=self.pos))
                    self.advance()

            elif self.current_char == '<':
                if self.peek() == '=':
                    tokens.append(Token(TT_LTE, pos_start=self.pos))
                    self.advance(2)
                else:
                    tokens.append(Token(TT_LT, pos_start=self.pos))
                    self.advance()

            elif self.current_char == '&':
                tokens.append(Token(TT_AND, pos_start=self.pos))
                self.advance()

            elif self.current_char == '|' and self.peek() == '|':
                tokens.append(Token(TT_OR, pos_start=self.pos))
                self.advance(2)

            elif self.current_char == "'":
                tokens.append(Token(TT_SNGLQTE, pos_start=self.pos))
                self.advance()

            elif self.current_char == '"':
                tokens.append(Token(TT_DBLQTE, pos_start=self.pos))
                self.advance()

            elif self.current_char == '\n':
                tokens.append(Token(TT_NEWLINE, pos_start=self.pos))
                self.advance()

            else:
                pos_start = self.pos.copy()
                char = self.current_char
                self.advance()
                return [], IllegalCharError(pos_start, self.pos, "'" + char + "'")

        tokens.append(Token(TT_EOF, pos_start=self.pos))
        return tokens, None

    def make_number(self):
        num_str = ''
        dot_count = 0
        pos_start = self.pos.copy()

        while self.current_char != None and self.current_char in DIGITS + '.':
            if self.current_char == '.':
                if dot_count == 1:
                    break
                dot_count += 1
                num_str += '.'
            else:
                num_str += self.current_char
            self.advance()
        
        if dot_count == 0:
            return Token(TT_INT, int(num_str), pos_start, self.pos)
        else:
            return Token(TT_FLOAT, float(num_str), pos_start, self.pos) 
        
    def make_variable(self):
        var_str = ''
        pos_start = self.pos.copy()

        while self.current_char != None and self.current_char != ' ':
            var_str += self.current_char
            self.advance()

        return Token(TT_VAR, var_str, pos_start, self.pos)
        
#################################
# NODES
#################################
        
class NumberNode:
    def __init__(self, tok):
        self.tok = tok

        self.pos_start = self.tok.pos_start
        self.pos_end = self.tok.pos_end
    
    def __repr__(self):
        return f'{self.tok}'
    
class BinOpNode:
    def __init__(self, left_node, op_tok, right_node):
        self.left_node = left_node
        self.op_tok = op_tok
        self.right_node = right_node

        self.pos_start = self.left_node.pos_start
        self.pos_end = self.right_node.pos_end

    def __repr__(self):
        return f'({self.left_node}, {self.op_tok}, {self.right_node})'
    
class UnaryOpNode:
    def __init__(self, op_tok, node):
        self.op_tok = op_tok
        self.node = node

        self.pos_start = self.op_tok.pos_start
        self.pos_end = node.pos_end

    def __repr__(self):
        return f'({self.op_tok}, {self.node})'
    
class MinNode:
    def __init__(self, left_node, right_node):
        self.left_node = left_node
        self.right_node = right_node

        self.pos_start = self.left_node.pos_start
        self.pos_end = self.right_node.pos_end

    def __repr__(self):
        return f'Min({self.left_node}, {self.right_node})'
    
class MaxNode:
    def __init__(self, left_node, right_node):
        self.left_node = left_node
        self.right_node = right_node

        self.pos_start = self.left_node.pos_start
        self.pos_end = self.right_node.pos_end

    def __repr__(self):
        return f'Max({self.left_node}, {self.right_node})'
    
class LogicalOpNode:
    def __init__(self, left_node, op_tok, right_node):
        self.left_node = left_node
        self.op_tok = op_tok
        self.right_node = right_node

        self.pos_start = self.left_node.pos_start
        self.pos_end = self.right_node.pos_end

    def __repr__(self):
        return f'({self.left_node} {self.op_tok} {self.right_node})'

class UnaryLogicalOpNode:
    def __init__(self, op_tok, node):
        self.op_tok = op_tok
        self.node = node

        self.pos_start = self.node.pos_start
        self.pos_end = self.node.pos_end

    def __repr__(self):
        return f'({self.op_tok} {self.node})'

class CompOpNode:
    def __init__(self, left_node, op_tok, right_node):
        self.left_node = left_node
        self.op_tok = op_tok
        self.right_node = right_node

        self.pos_start = self.left_node.pos_start
        self.pos_end = self.right_node.pos_end

    def __repr__(self):
        return f'({self.left_node} {self.op_tok} {self.right_node})'
    
class StatementsNode:
    def __init__(self, statements):
        self.statements = statements
        self.pos_start = statements[0].pos_start if len(statements) > 0 else None
        self.pos_end = statements[-1].pos_end if len(statements) > 0 else None

    def __repr__(self):
        return f"{self.statements}"
    
class VariableNode:
    def __init__(self, var_name, value):
        self.var_name = var_name
        self.value = value
    
        self.pos_start = self.var_name.pos_start
        self.pos_end = self.var_name.pos_end if value is None else self.value.pos_end
    
    def __repr__(self):
        return f'({self.var_name} = {self.value})'

#################################
# PARSER RESULT
#################################
    
class ParseResult:
    def __init__(self):
        self.error = None
        self.node = None

    def register(self, res):
        if isinstance(res, ParseResult):
            if res.error:
                self.error = res.error
            return res.node

        return res
    
    def success(self, node):
        self.node = node
        return self

    def failure(self, error):
        self.error = error
        return self

#################################
# PARSER
#################################

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.tok_idx = -1
        self.advance()

    def advance(self):
        self.tok_idx += 1
        if self.tok_idx < len(self.tokens):
            self.current_tok = self.tokens[self.tok_idx]
        return self.current_tok

    def parse(self):
        res = self.statements()
        if not res.error and self.current_tok.type != TT_EOF:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected '+', '-', '*', '/' or comparison operator"
            ))
        return res

    def statements(self):
        res = ParseResult()
        statements = []

        while self.current_tok.type != TT_EOF:
            stmt = res.register(self.statement())
            if res.error:
                return res
            statements.append(stmt)

            if self.current_tok.type == TT_NEWLINE:
                res.register(self.advance())  # Advance past the newline
            elif self.current_tok.type != TT_EOF:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected 'NEWLINE'"
                ))

        return res.success(StatementsNode(statements))

    def statement(self):
        return self.expr()

    def factor(self):
        res = ParseResult()
        tok = self.current_tok

        if tok.type in (TT_PLUS, TT_MINUS):
            res.register(self.advance())
            factor = res.register(self.factor()) 
            if res.error:
                return res
            return res.success(UnaryOpNode(tok, factor))
        
        elif tok.type == TT_NOT:
            res.register(self.advance())
            factor = res.register(self.factor())
            if res.error:
                return res
            return res.success(UnaryLogicalOpNode(tok, factor))

        elif tok.type in (TT_INT, TT_FLOAT):
            res.register(self.advance())
            return res.success(NumberNode(tok))
        
        elif tok.type == TT_LPAREN:
            res.register(self.advance())
            expr = res.register(self.expr())
            if res.error:
                return res
            if self.current_tok.type == TT_RPAREN:
                res.register(self.advance())
                return res.success(expr) 
            else:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected ')' "
                ))

        elif tok.type in (TT_MIN, TT_MAX):
            tok_str = 'Min' if tok.type == TT_MIN else 'Max'
            res.register(self.advance())
            if self.current_tok.type != TT_LPAREN:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    f"Expected '(' after '{tok_str}'"
                ))
            
            res.register(self.advance())  # Skip the '(' char
            left_expr = res.register(self.expr())
            if res.error:
                return res
            
            if self.current_tok.type != TT_COMMA:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    f"Expected ',' after first argument in '{tok_str}'"
                ))
            
            res.register(self.advance())  # Skip the ',' char
            right_expr = res.register(self.expr())
            if res.error:
                return res
            
            if self.current_tok.type != TT_RPAREN:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    f"Expected ')' after second argument in '{tok_str}'"
                ))

            res.register(self.advance())
            return res.success(
                MinNode(left_expr, right_expr)) if tok.type == TT_MIN else res.success(
                    MaxNode(left_expr, right_expr))
        
        elif tok.type in (TT_VAR):
            var_name = tok
            res.register(self.advance())

            # Check if it's a variable assignment
            if self.current_tok.type == TT_ASSIGN:
                res.register(self.advance())
                expr_value = res.register(self.expr())  # Parse the value as an expression
                if res.error:
                    return res
                return res.success(VariableNode(var_name, expr_value))
            # Otherwise, it's a variable usage
            else:
                return res.success(VariableNode(var_name, None))

        return res.failure(InvalidSyntaxError(
            tok.pos_start, tok.pos_end,
            "Expected int or float"
        ))

    def comp_expr(self):
        return self.comp_op(self.arith_expr, (TT_EQUAL, TT_NEQUAL, TT_GT, TT_LT, TT_GTE, TT_LTE))

    def expr(self):
        return self.bin_op(self.logic_expr, (TT_OR, TT_AND))

    def term(self):
        return self.bin_op(self.factor, (TT_MUL, TT_DIV, TT_POWER))
    
    def logic_expr(self):
        return self.logical_op(self.comp_expr, (TT_AND, TT_OR))
    
    def arith_expr(self):
        return self.bin_op(self.term, (TT_PLUS, TT_MINUS))

    def bin_op(self, func, ops):
        res = ParseResult()
        left = res.register(func())
        if res.error:
            return res

        while self.current_tok.type in ops:
            op_tok = self.current_tok
            res.register(self.advance())
            right = res.register(func())
            if res.error:
                return res
            left = BinOpNode(left, op_tok, right)
        
        return res.success(left)     

    def logical_op(self, func, ops):
        res = ParseResult()
        left = res.register(func())
        if res.error:
            return res

        while self.current_tok.type in ops:
            op_tok = self.current_tok
            res.register(self.advance())
            right = res.register(func())
            if res.error:
                return res
            left = LogicalOpNode(left, op_tok, right)
        
        return res.success(left)
    
    def comp_op(self, func, ops):
        res = ParseResult()
        left = res.register(func())
        if res.error:
            return res

        if self.current_tok.type in ops:
            op_tok = self.current_tok
            res.register(self.advance())
            right = res.register(func())
            if res.error:
                return res
            left = CompOpNode(left, op_tok, right)
        
        return res.success(left)

#################################
# RUNTIME RESULT
#################################
    
class RTResult:
    def __init__(self):
        self.value = None
        self.error = None

    def register(self, res):
        if res.error:
            self.error = res.error
        return res.value
    
    def success(self, value):
        self.value = value
        return self
    
    def failure(self, error):
        self.error = error
        return self

#################################
# VALUES
#################################

class Number:
    def __init__(self, value):
        self.value = value
        self.set_pos()
        self.set_context()

    def set_pos(self, pos_start=None, pos_end=None):
        self.pos_start = pos_start
        self.pos_end = pos_end
        return self
    
    def set_context(self, context=None):
        self.context = context
        return self

    def added_to(self, other):
        if isinstance(other, Number):
            return Number(self.value + other.value).set_context(self.context), None
        
    def subbed_by(self, other):
        if isinstance(other, Number):
            return Number(self.value - other.value).set_context(self.context), None
    
    def powed_by(self, other):
        if isinstance(other, Number):
            return Number(self.value ** other.value).set_context(self.context), None
        
    def multed_by(self, other):
        if isinstance(other, Number):
            return Number(self.value * other.value).set_context(self.context), None
        
    def dived_by(self, other):
        if isinstance(other, Number):
            if other.value == 0:
                return None, RTError(
                    other.pos_start, other.pos_end,
                    "Division by zero is undefined\n",
                    self.context
                )
            
            return Number(self.value / other.value).set_context(self.context), None
        
    def __str__(self):
        if isinstance(self.value, bool):
            return str(self.value)
        return str(self.value)
    
class CompOp:
    def __init__(self, value):
        self.value = value
        self.set_pos()
        self.set_context()

    def set_pos(self, pos_start=None, pos_end=None):
        self.pos_start = pos_start
        self.pos_end = pos_end
        return self
    
    def set_context(self, context=None):
        self.context = context
        return self
    
    def not_this(self):
        return CompOp(not self.value).set_context(self.context), None
    
    def equal_to(self, other):
        if isinstance(other, CompOp):
            return CompOp(self.value == other.value).set_context(self.context), None
        
    def not_equal_to(self, other):
        if isinstance(other, CompOp):
            return CompOp(self.value != other.value).set_context(self.context), None

    def greater_than(self, other):
        if isinstance(other, CompOp):
            return CompOp(self.value > other.value).set_context(self.context), None

    def less_than(self, other):
        if isinstance(other, CompOp):
            return CompOp(self.value < other.value).set_context(self.context), None

    def greater_than_or_equal_to(self, other):
        if isinstance(other, CompOp):
            return CompOp(self.value >= other.value).set_context(self.context), None

    def less_than_or_equal_to(self, other):
        if isinstance(other, CompOp):
            return CompOp(self.value <= other.value).set_context(self.context), None

    def __str__(self):
        return str(self.value)

class LogicalOp:
    def __init__(self, value):
        self.value = bool(value)
        self.set_pos()
        self.set_context()

    def set_pos(self, pos_start=None, pos_end=None):
        self.pos_start = pos_start
        self.pos_end = pos_end
        return self
    
    def set_context(self, context=None):
        self.context = context
        return self
    
    def not_this(self):
        return LogicalOp(not self.value).set_context(self.context), None 

    def and_with(self, other):
        if isinstance(other, (LogicalOp, Number)):
            return LogicalOp(self.value and bool(other.value)).set_context(self.context), None

    def or_with(self, other):
        if isinstance(other, (LogicalOp, Number)):
            return LogicalOp(self.value or bool(other.value)).set_context(self.context), None

    def __str__(self):
        return str(self.value)  # Convert back to int for display
    
class Variable:
    def __init__(self, value=None):
        self.set_pos()
        self.set_context()
        self.value = value
        self.variables = {}  # Dictionary to store variables by name

    def set_pos(self, pos_start=None, pos_end=None):
        self.pos_start = pos_start
        self.pos_end = pos_end
        return self
    
    def set_context(self, context=None):
        self.context = context
        return self

    def assign_to(self, name, other):
        if isinstance(other, Number):
            # Assign the value of 'other' to the var_name name
            self.variables[name] = Number(other.value).set_context(self.context)
            return self.variables[name], None
        else:
            # Handle other types or raise an error
            return None, ValueError("Assignment only supports Numbers.")

    def get_value(self, name):
        # Retrieve the value of a var_name by name
        return self.variables.get(name, None)
    
#################################
# CONTEXT
#################################
    
class Context:
    def __init__(self, display_name, parent=None, parent_entry_pos=None):
        self.display_name = display_name
        self.parent = parent
        self.parent_entry_pos = parent_entry_pos
        self.variables = {}  # Dictionary to store variables

#################################
# INTERPRETER
#################################
    
class Interpreter:
    def visit(self, node, context):
        method_name = f'visit_{type(node).__name__}' # Creates visit_BinOpNode or visit_NumberNode
        method = getattr(self, method_name, self.no_visit_method)
        return method(node, context)
    
    def no_visit_method(self, node, context):
        raise Exception(f'No visit_{type(node).__name__} method defined')
    
    #################################

    def visit_NumberNode(self, node, context):
        return RTResult().success(Number(node.tok.value).set_context(context).set_pos(node.pos_start, node.pos_end))

    def visit_BinOpNode(self, node, context):
        res = RTResult()
        left = res.register(self.visit(node.left_node, context))
        if res.error:
            return res
        right = res.register(self.visit(node.right_node, context))
        if res.error:
            return res

        result = None  # do i need this?

        error = None  # Initialize error to None

        if node.op_tok.type == TT_PLUS:
            result, error = left.added_to(right)
        elif node.op_tok.type == TT_MINUS:
            result, error = left.subbed_by(right)
        elif node.op_tok.type == TT_MUL:
            result, error = left.multed_by(right)
        elif node.op_tok.type == TT_DIV:
            result, error = left.dived_by(right)
        elif node.op_tok.type == TT_POWER:
            result, error = left.powed_by(right)
        
        if error:
            return res.failure(error)
        elif result is not None:
            return res.success(result.set_pos(node.pos_start, node.pos_end))
        else:
            return res.failure(InvalidSyntaxError(
                node.pos_start, node.pos_end,
                f"Operation failed unexpectedly with operator '{node.op_tok.type}'\n"
            ))

    def visit_UnaryOpNode(self, node, context):
        res = RTResult()
        number = res.register(self.visit(node.node, context))
        if res.error:
            return res

        error = None  # Initialize error to None

        if node.op_tok.type == TT_MINUS:
            number, error = number.multed_by(Number(-1))

        if error:
            return res.failure(error)
        else:
            return res.success(number.set_pos(node.pos_start, node.pos_end))

    def visit_MinNode(self, node, context):
        res = RTResult()
        left = res.register(self.visit(node.left_node, context))
        if res.error:
            return res
        right = res.register(self.visit(node.right_node, context))
        if res.error:
            return res

        min_value = min(left.value, right.value)
        return res.success(Number(min_value).set_context(context).set_pos(node.pos_start, node.pos_end))
    
    def visit_MaxNode(self, node, context):
        res = RTResult()
        left = res.register(self.visit(node.left_node, context))
        if res.error:
            return res
        right = res.register(self.visit(node.right_node, context))
        if res.error:
            return res

        max_value = max(left.value, right.value)
        return res.success(Number(max_value).set_context(context).set_pos(node.pos_start, node.pos_end))
    
    def visit_UnaryLogicalOpNode(self, node, context):
        res = RTResult()
        operand = res.register(self.visit(node.node, context))
        if res.error:
            return res

        if node.op_tok.type == TT_NOT:
            if hasattr(operand, 'not_this'):
                result, error = operand.not_this()
            else:
                result, error = LogicalOp(operand.value).not_this()
            
            if error:
                return res.failure(error)
            return res.success(result.set_pos(node.pos_start, node.pos_end))
        
        return res.failure(InvalidSyntaxError(
            node.pos_start, node.pos_end,
            f"Invalid unary operator '{node.op_tok.type}'"
        ))
        
    def visit_LogicalOpNode(self, node, context):
        res = RTResult()
        left = res.register(self.visit(node.left_node, context))
        if res.error: 
            return res
        right = res.register(self.visit(node.right_node, context))
        if res.error: 
            return res

        if node.op_tok.type == TT_AND:
            result, error = LogicalOp(left.value).and_with(LogicalOp(right.value))
        elif node.op_tok.type == TT_OR:
            result, error = LogicalOp(left.value).or_with(LogicalOp(right.value))
        else:
            return res.failure(InvalidSyntaxError(
                node.pos_start, node.pos_end,
                f"Invalid logical operator '{node.op_tok.type}'"
            ))

        if error:
            return res.failure(error)
        else:
            return res.success(result.set_pos(node.pos_start, node.pos_end))
        
    def visit_CompOpNode(self, node, context):
        res = RTResult()
        left = res.register(self.visit(node.left_node, context))
        if res.error:
            return res
        right = res.register(self.visit(node.right_node, context))
        if res.error:
            return res

        if isinstance(left, Number) and isinstance(right, Number):
            left = CompOp(left.value).set_context(context).set_pos(node.pos_start, node.pos_end)
            right = CompOp(right.value).set_context(context).set_pos(node.pos_start, node.pos_end)

        if node.op_tok.type == TT_GT:
            result, error = left.greater_than(right)
        elif node.op_tok.type == TT_LT:
            result, error = left.less_than(right)
        elif node.op_tok.type == TT_GTE:
            result, error = left.greater_than_or_equal_to(right)
        elif node.op_tok.type == TT_LTE:
            result, error = left.less_than_or_equal_to(right)
        elif node.op_tok.type == TT_EQUAL:
            result, error = left.equal_to(right)
        elif node.op_tok.type == TT_NEQUAL:
            result, error = left.not_equal_to(right)
        
        if error:
            return res.failure(error)
        elif result is not None:
            return res.success(result.set_pos(node.pos_start, node.pos_end))
        else:
            return res.failure(InvalidSyntaxError(
                node.pos_start, node.pos_end,
                f"Operation failed unexpectedly with operator '{node.op_tok.type}'\n"
            ))
        
    def visit_StatementsNode(self, node, context):
        res = RTResult()
        results = []        # Just for debugging
        for statement in node.statements:
            value = res.register(self.visit(statement, context))
            if res.error:
                return res
            results.append(value)
        return res.success(results)
    
    def visit_VariableNode(self, node, context):
        res = RTResult()

        # Get the variable name directly from the token
        variable_name = node.var_name.value

        if node.value:
            # Variable assignment
            value = res.register(self.visit(node.value, context))
            if res.error:
                return res
            context.variables[variable_name] = value
            return res.success(value)
        else:
            # Variable usage
            if variable_name in context.variables:
                return res.success(context.variables[variable_name])
            else:
                return res.failure(RTError(
                    node.pos_start, node.pos_end,
                    f"'{variable_name}' is not defined",
                    context
                ))

#################################
# RUN
#################################
        
def run(filename, text):
    # Generate tokens
    lexer = Lexer(filename, text)
    tokens, error = lexer.make_tokens()
    if error: 
        return None, error
    
    # Generate AST
    parser = Parser(tokens)
    ast = parser.parse()
    if ast.error: 
        return None, ast.error

    # Run program
    interpreter = Interpreter()
    context = Context('<program>')
    result = interpreter.visit(ast.node, context)
    
    if result.error:
        return None, result.error
    
    # Print the result of each statement for debugging
    for statement_result in result.value:
        if statement_result is not None:
            print(statement_result)
    
    return result.value, None