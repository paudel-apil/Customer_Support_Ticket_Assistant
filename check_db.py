import sqlite3

with sqlite3.connect('tickets.db') as conn:
    for row in conn.execute("SELECT id, customer_id, title, department, priority, created_at FROM tickets"):
        print(row)