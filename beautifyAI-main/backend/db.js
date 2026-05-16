// Mock DB to bypass better-sqlite3 C++ compilation issues on Node 24
const users = [];
let nextId = 1;

const db = {
  prepare: (sql) => {
    return {
      get: (...args) => {
        if (sql.includes('SELECT * FROM users WHERE email = ?')) {
          return users.find(u => u.email === args[0]);
        }
        if (sql.includes('SELECT id FROM users WHERE email = ?')) {
          const user = users.find(u => u.email === args[0]);
          return user ? { id: user.id } : undefined;
        }
        if (sql.includes('SELECT id FROM users WHERE email = ? AND reset_token = ? AND reset_token_expiry > ?')) {
          const user = users.find(u => u.email === args[0] && u.reset_token === args[1] && new Date(u.reset_token_expiry) > new Date(args[2]));
          return user ? { id: user.id } : undefined;
        }
        return undefined;
      },
      run: (...args) => {
        if (sql.includes('INSERT INTO users')) {
          const user = {
            id: nextId++,
            name: args[0],
            email: args[1],
            password_hash: args[2],
            reset_token: null,
            reset_token_expiry: null
          };
          users.push(user);
          return { lastInsertRowid: user.id };
        }
        if (sql.includes('UPDATE users SET reset_token = ?, reset_token_expiry = ? WHERE id = ?')) {
          const user = users.find(u => u.id === args[2]);
          if (user) {
            user.reset_token = args[0];
            user.reset_token_expiry = args[1];
          }
          return { changes: 1 };
        }
        if (sql.includes('UPDATE users SET password_hash = ?, reset_token = NULL, reset_token_expiry = NULL WHERE id = ?')) {
          const user = users.find(u => u.id === args[1]);
          if (user) {
            user.password_hash = args[0];
            user.reset_token = null;
            user.reset_token_expiry = null;
          }
          return { changes: 1 };
        }
        return { changes: 0 };
      }
    };
  }
};

module.exports = db;
