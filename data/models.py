

class Position:
    ticker = str
    quantity = 0
    price = 0

    def __init__(self, ticker, quantity, price):
        self.ticker = ticker
        self.quantity = quantity
        self.price = price


class Portfolio:
    positions = {str: Position}
    equity = 0
    cash = 0
    total = equity + cash

    def __init__(self, cash):
        self.cash = cash
        self.total = cash
    
    def add_position(self, symbol, quantity, price):
        new_position = Position(symbol, quantity, price)
        self.positions[symbol] = new_position
        self.cash -= quantity * price
        self.total = self.cash + self.equity

    def remove_position(self, symbol, quantity, price):
        self.positions[symbol]['quantity'] -= quantity
        if self.positions[symbol]['quantity'] == 0:
            del self.positions[symbol]
        self.cash += quantity * price
        self.total = self.cash + self.equity

    def update_position(self, symbol, quantity, price):
        self.positions[symbol]['quantity'] = quantity
        self.positions[symbol]['price'] = price
        self.total = self.cash + self.equity

    def update_portfolio(self, equity):
        self.equity = equity
        self.total = self.cash + self.equity

    def get_portfolio(self):
        return self.positions, self.cash, self.equity, self.total
    
