"""CRUD operations and query builder.

NOTE: For high-performance queries on large tables (market_candles, etc.),
use optimized queries with specific column selection instead of SELECT *.
See: MarketCandleQueries for market_candle specific queries.
"""
from typing import Any, List, Tuple


class QueryBuilder:
    """Helper for building SQL queries.
    
    NOTE: For high-performance queries on large tables, use optimized queries
    with specific column selection instead of SELECT *.
    """
    
    def __init__(self, table: str):
        self.table = table
        self.conditions: List[str] = []
        self.args: List[Any] = []
        self.limit: int | None = None
        self.offset: int | None = None
        self.order_by: str | None = None
    
    def where(self, condition: str, *args: Any) -> "QueryBuilder":
        """Add a WHERE condition.
        
        Args:
            condition: WHERE clause condition
            *args: Arguments for the condition
            
        Returns:
            QueryBuilder for chaining
        """
        self.conditions.append(condition)
        self.args.extend(args)
        return self
    
    def limit(self, limit: int) -> "QueryBuilder":
        """Add a LIMIT clause.
        
        Args:
            limit: Limit value
            
        Returns:
            QueryBuilder for chaining
        """
        self.limit = limit
        return self
    
    def offset(self, offset: int) -> "QueryBuilder":
        """Add an OFFSET clause.
        
        Args:
            offset: Offset value
            
        Returns:
            QueryBuilder for chaining
        """
        self.offset = offset
        return self
    
    def order_by(self, order_by: str) -> "QueryBuilder":
        """Add an ORDER BY clause.
        
        Args:
            order_by: ORDER BY clause
            
        Returns:
            QueryBuilder for chaining
        """
        self.order_by = order_by
        return self
    
    def build(self) -> Tuple[str, List[Any]]:
        """Build the final SQL query.
        
        Returns:
            Tuple of (SQL query, arguments)
        """
        sql = f"SELECT * FROM {self.table}"
        
        if self.conditions:
            sql += " WHERE " + " AND ".join(self.conditions)
        
        if self.order_by:
            sql += f" ORDER BY {self.order_by}"
        
        if self.limit is not None:
            sql += f" LIMIT {self.limit}"
        
        if self.offset is not None:
            sql += f" OFFSET {self.offset}"
        
        return sql, self.args
    
    def build_count(self) -> Tuple[str, List[Any]]:
        """Build a COUNT query.
        
        Returns:
            Tuple of (COUNT query, arguments)
        """
        sql = f"SELECT COUNT(*) FROM {self.table}"
        
        if self.conditions:
            sql += " WHERE " + " AND ".join(self.conditions)
        
        return sql, self.args
    
    def build_exists(self) -> Tuple[str, List[Any]]:
        """Build an EXISTS query.
        
        Returns:
            Tuple of (EXISTS query, arguments)
        """
        where_clause = ""
        if self.conditions:
            where_clause = " WHERE " + " AND ".join(self.conditions)
        
        sql = f"SELECT EXISTS(SELECT 1 FROM {self.table}{where_clause})"
        return sql, self.args


def build_insert(table: str, columns: List[str]) -> str:
    """Build an INSERT query.
    
    Args:
        table: Table name
        columns: Column names
        
    Returns:
        INSERT SQL query
    """
    placeholders = [f"${i + 1}" for i in range(len(columns))]
    return f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"


def build_update(table: str, set_clauses: List[str], where_clause: str) -> str:
    """Build an UPDATE query.
    
    Args:
        table: Table name
        set_clauses: SET clauses
        where_clause: WHERE clause
        
    Returns:
        UPDATE SQL query
    """
    return f"UPDATE {table} SET {', '.join(set_clauses)} WHERE {where_clause}"


def build_delete(table: str, where_clause: str) -> str:
    """Build a DELETE query.
    
    Args:
        table: Table name
        where_clause: WHERE clause
        
    Returns:
        DELETE SQL query
    """
    return f"DELETE FROM {table} WHERE {where_clause}"
