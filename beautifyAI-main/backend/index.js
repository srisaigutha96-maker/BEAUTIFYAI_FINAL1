require('dotenv').config();
const express = require('express');
const cors = require('cors');
const bcrypt = require('bcryptjs');
const crypto = require('crypto');
const nodemailer = require('nodemailer');
const db = require('./db');
const path = require('path');
const fs = require('fs');
const { exec, spawn } = require('child_process');
const multer = require('multer');
const axios = require('axios');
const { v4: uuidv4 } = require('uuid');

const app = express();
const PORT = 3001;

// ── Configuration ─────────────────────────────────────────────────────
const TEMP_DIR = path.join(__dirname, 'temp_images');
const MODEL_DIR = path.join(__dirname, 'model');
const CHECKPOINT_PATH = 'checkpoint/checkpoint_epoch_24.pth';

if (!fs.existsSync(TEMP_DIR)) {
  fs.mkdirSync(TEMP_DIR, { recursive: true });
}

// ── Background Cleanup Task ───────────────────────────────────────────
// Remove stale temp images to preserve user privacy
const cleanTempImages = () => {
  try {
    const files = fs.readdirSync(TEMP_DIR);
    const now = Date.now();
    files.forEach(file => {
      const filePath = path.join(TEMP_DIR, file);
      const stats = fs.statSync(filePath);
      if (now - stats.mtimeMs > 3600000) { // older than 1 hour
        fs.unlinkSync(filePath);
        console.log(`[BeautifyAI] 🧹 Cleared stale temp image: ${file}`);
      }
    });
  } catch (err) {
    console.error('[BeautifyAI] ⚠️ Background cleanup error:', err.message);
  }
};

// Run immediately on startup
cleanTempImages();

// And periodically every hour
setInterval(cleanTempImages, 3600000);

// ── Middleware ────────────────────────────────────────────────────────
app.use(cors({
  origin: [
    'http://localhost:5173',
    'http://localhost:5174',
    'http://localhost:3000',
    'http://127.0.0.1:3000',
    'https://beautifyai-final-1-gsfr.vercel.app',
    'https://beautifyai-final-1-gsfr-srisaigutha96-makers-projects.vercel.app'
  ],
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS']
}));
app.use(express.json());

// ── Email Configuration ───────────────────────────────────────────────
let transporter;

async function initMailer() {
  if (process.env.SMTP_USER) {
    const smtpPort = parseInt(process.env.SMTP_PORT || '465');
    transporter = nodemailer.createTransport({
      host: process.env.SMTP_HOST || 'smtp.gmail.com',
      port: smtpPort,
      secure: smtpPort === 465,
      auth: {
        user: process.env.SMTP_USER,
        pass: process.env.SMTP_PASS,
      },
    });
    console.log("[BeautifyAI] 📧 SMTP configured with provided credentials.");
  } else {
    console.log("[BeautifyAI] ⚠️ No SMTP_USER found in .env. Creating Ethereal test account for development...");
    const testAccount = await nodemailer.createTestAccount();
    transporter = nodemailer.createTransport({
      host: "smtp.ethereal.email",
      port: 587,
      secure: false,
      auth: {
        user: testAccount.user,
        pass: testAccount.pass,
      },
    });
    console.log("[BeautifyAI] 📧 Ethereal test SMTP configured.");
  }
}

initMailer();

// Set up multer for image uploads
const upload = multer({ dest: TEMP_DIR });

// ── Helpers ───────────────────────────────────────────────────────────
const _safeRemove = (filePath) => {
  try {
    if (fs.existsSync(filePath)) fs.unlinkSync(filePath);
  } catch (err) {
    console.warn(`[BeautifyAI] Could not delete temp file: ${filePath}`, err);
  }
};

const _base64Encode = (filePath) => {
  const bitmap = fs.readFileSync(filePath);
  return Buffer.from(bitmap).toString('base64');
};

const _fallbackResponse = (res, inputPath, message) => {
  try {
    const encoded = _base64Encode(inputPath);
    const ext = path.extname(inputPath).toLowerCase();
    const mimeType = ext === '.png' ? 'image/png' : 'image/jpeg';
    
    _safeRemove(inputPath);
    
    return res.json({
      success: false,
      fallback: true,
      message,
      image_url: `data:${mimeType};base64,${encoded}`,
      status: 'fallback'
    });
  } catch (err) {
    _safeRemove(inputPath);
    console.error('[BeautifyAI] Fatal error in fallback:', err);
    return res.status(500).json({ success: false, message: 'Critical error serving fallback image.' });
  }
};

// ── Health Check ──────────────────────────────────────────────────────
app.get('/health', (req, res) => {
  const checkpointFull = path.join(MODEL_DIR, CHECKPOINT_PATH);
  const modelExists = fs.existsSync(checkpointFull);
  res.json({ 
    status: 'ok', 
    message: 'BeautifyAI Unified Backend is running ✅',
    model_ready: modelExists,
    port: PORT
  });
});

app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', message: 'BeautifyAI Auth API is running ✅' });
});

// ── AI Inference Endpoint ─────────────────────────────────────────────
app.post('/beautify', upload.single('image'), async (req, res) => {
  console.log(`[BeautifyAI] 📩 Received BEAUTIFY request from ${req.ip}`);
  
  if (!req.file) {
    return res.status(400).json({ success: false, message: 'No image uploaded.' });
  }

  const intensity = req.body.intensity || 0.5;
  const inputPath = req.file.path;
  const outputFileName = `enhanced_${uuidv4()}.jpg`;
  const outputPath = path.join(TEMP_DIR, outputFileName);
  
  console.log(`[BeautifyAI]    Image: ${req.file.originalname}, Intensity: ${intensity}`);

  // Check if model exists
  const checkpointFull = path.join(MODEL_DIR, CHECKPOINT_PATH);
  if (!fs.existsSync(checkpointFull)) {
    console.warn(`[BeautifyAI] ⚠️ Model not found at ${checkpointFull}`);
    return _fallbackResponse(res, inputPath, `Model checkpoint not found at ${CHECKPOINT_PATH}. Showing original.`);
  }

  // EXACT COMMAND: python inference.py -i [input] -o [output] -intensity [val] -w checkpoint/checkpoint_epoch_24.pth
  const cmd = `python inference.py -i "${inputPath}" -o "${outputPath}" -intensity ${intensity} -w "${CHECKPOINT_PATH}"`;
  
  console.log(`[BeautifyAI] 🚀 Launching inference: ${cmd}`);

  exec(cmd, { cwd: MODEL_DIR, timeout: 300000 }, (error, stdout, stderr) => {
    if (error) {
      console.error(`[BeautifyAI] ❌ Inference failed: ${error.message}`);
      console.error(`[BeautifyAI]    stderr: ${stderr}`);
      
      if (stdout.includes("NO_FACE_DETECTED") || stderr.includes("NO_FACE_DETECTED")) {
        _safeRemove(inputPath);
        return res.status(400).json({ success: false, message: 'No human face detected in the image. Please upload a clear photo of a person.' });
      }
      
      return _fallbackResponse(res, inputPath, `AI inference failed: ${error.message}`);
    }

    if (!fs.existsSync(outputPath) || fs.statSync(outputPath).size === 0) {
      console.error(`[BeautifyAI] ❌ Output file missing or empty.`);
      return _fallbackResponse(res, inputPath, "AI model failed to generate output.");
    }

    try {
      console.log(`[BeautifyAI] ✅ Inference succeeded. Output: ${outputFileName}`);
      const encoded = _base64Encode(outputPath);
      
      _safeRemove(inputPath);
      _safeRemove(outputPath);

      return res.json({
        success: true,
        fallback: false,
        message: `AI beautification applied at intensity ${intensity}`,
        image_url: `data:image/jpeg;base64,${encoded}`,
        status: 'success'
      });
    } catch (err) {
      console.error(`[BeautifyAI] ❌ Error processing success response: ${err.message}`);
      return _fallbackResponse(res, inputPath, "Error serving enhanced image.");
    }
  });
});

// ── CodeFormer API Beautify Endpoint ──────────────────────────────────
app.post('/api/beautify', upload.single('image'), (req, res) => {
  console.log(`[BeautifyAI] 📩 Received API BEAUTIFY request from ${req.ip}`);
  if (!req.file) {
    return res.status(400).json({ success: false, message: 'No image uploaded.' });
  }

  const beforeFileName = `before_${uuidv4()}.jpg`;
  const afterFileName = `after_${uuidv4()}.jpg`;
  const beforePath = path.join(TEMP_DIR, beforeFileName);
  const afterPath = path.join(TEMP_DIR, afterFileName);

  // Maintain Before and After versions in temp_images
  fs.copyFileSync(req.file.path, beforePath);

  const intensity = req.body.intensity || 0.5;
  const child = spawn('python', ['main.py', '-i', req.file.path, '-o', afterPath, '-int', intensity], { cwd: __dirname });

  let outData = '';
  child.stdout.on('data', data => { outData += data; console.log(`[CodeFormer]: ${data}`) });
  child.stderr.on('data', data => { outData += data; console.error(`[CodeFormer Error]: ${data}`) });

  child.on('error', (err) => {
    console.error(`[BeautifyAI] ❌ Failed to start CodeFormer process: ${err.message}`);
    _safeRemove(req.file.path);
    _safeRemove(beforePath);
    _safeRemove(afterPath);
    // Don't res.status(500) if headers already sent, but let's assume they aren't
    if (!res.headersSent) {
      return res.status(500).json({ success: false, message: 'Failed to start AI engine.' });
    }
  });

  child.on('close', (code) => {
    _safeRemove(req.file.path); // Remove multer initial chunk

    if (outData.includes("NO_FACE_DETECTED")) {
      _safeRemove(beforePath);
      _safeRemove(afterPath);
      return res.status(400).json({ success: false, message: 'No human face detected in the image. Please upload a clear photo of a person.' });
    }

    if (code !== 0 || !fs.existsSync(afterPath)) {
      console.warn(`[BeautifyAI] ⚠️ CodeFormer failed (code ${code}). Serving Pillow-enhanced fallback.`);
      // Run Pillow fallback via inference.py so user still gets an enhanced image
      const modelDir = MODEL_DIR;
      const fbCmd = `python inference.py -i "${beforePath}" -o "${afterPath}" -intensity 0.5 -w "${CHECKPOINT_PATH}"`;
      exec(fbCmd, { cwd: modelDir, timeout: 60000 }, (fbErr) => {
        // Whether Pillow fallback worked or not, always serve something
        const afterExists = fs.existsSync(afterPath);
        try {
          const beforeEncoded = _base64Encode(beforePath);
          const afterEncoded = afterExists ? _base64Encode(afterPath) : beforeEncoded;
          _safeRemove(beforePath);
          if (afterExists) _safeRemove(afterPath);
          return res.json({
            success: true,
            fallback: true,
            message: 'Enhancement applied (fallback mode)',
            before_url: `data:image/jpeg;base64,${beforeEncoded}`,
            after_url:  `data:image/jpeg;base64,${afterEncoded}`
          });
        } catch (encErr) {
          _safeRemove(beforePath);
          if (afterExists) _safeRemove(afterPath);
          return res.status(500).json({ success: false, message: 'Error serving fallback image.' });
        }
      });
      return; // Exit early – response is handled in exec callback
    }

    try {
      const beforeEncoded = _base64Encode(beforePath);
      const afterEncoded = _base64Encode(afterPath);

      // Cleanup temporary files after they have been encoded to base64
      _safeRemove(beforePath);
      _safeRemove(afterPath);

      console.log(`[BeautifyAI] ✅ Processed successfully. Images removed from temp_images.`);
      
      return res.json({
        success: true,
        message: 'Skin beautification completed using CodeFormer',
        before_url: `data:image/jpeg;base64,${beforeEncoded}`,
        after_url: `data:image/jpeg;base64,${afterEncoded}`
      });
    } catch (err) {
      // Ensure we clean up even if an error occurs during encoding
      _safeRemove(beforePath);
      _safeRemove(afterPath);
      return res.status(500).json({ success: false, message: 'Error processing response' });
    }
  });
});

// ── Chat Proxy Endpoint ───────────────────────────────────────────────
app.post('/chat', async (req, res) => {
  console.log(`[BeautifyAI] 📩 Received CHAT request from ${req.ip}`);
  const OLLAMA_URL = "http://localhost:11434/api/chat";
  
  try {
    const response = await axios.post(OLLAMA_URL, req.body, { responseType: 'stream' });
    response.data.pipe(res);
  } catch (err) {
    console.warn("[BeautifyAI] ⚠️ Ollama not reachable or error in chat proxying.");
    res.status(503).json({ error: "Ollama assistant is currently unavailable." });
  }
});

// ── POST /api/signup ──────────────────────────────────────────────────
app.post('/api/signup', async (req, res) => {
  const { name = '', email, password } = req.body;

  // Validation
  if (!email || !email.includes('@')) {
    return res.status(400).json({ success: false, message: 'Please enter a valid email address.' });
  }
  if (!password || password.length < 8) {
    return res.status(400).json({ success: false, message: 'Password must be at least 8 characters.' });
  }

  // Check if email already exists
  const existing = db.prepare('SELECT id FROM users WHERE email = ?').get(email.toLowerCase().trim());
  if (existing) {
    return res.status(409).json({ success: false, message: 'An account with this email already exists.' });
  }

  // Hash the password with bcrypt (12 rounds)
  const password_hash = await bcrypt.hash(password, 12);

  // Insert new user
  const stmt = db.prepare(
    'INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)'
  );
  const result = stmt.run(name.trim(), email.toLowerCase().trim(), password_hash);

  console.log(`✅ New user signed up: ${email}`);

  return res.status(201).json({
    success: true,
    message: 'Account created successfully!',
    user: { id: result.lastInsertRowid, email: email.toLowerCase().trim(), name: name.trim() },
  });
});

// ── POST /api/login ───────────────────────────────────────────────────
app.post('/api/login', async (req, res) => {
  const { email, password } = req.body;

  // Validation
  if (!email || !password) {
    return res.status(400).json({ success: false, message: 'Email and password are required.' });
  }

  // Look up user by email
  const user = db.prepare('SELECT * FROM users WHERE email = ?').get(email.toLowerCase().trim());
  if (!user) {
    console.log(`❌ Login failed – email not found: ${email}`);
    return res.status(401).json({ success: false, message: 'No account found with this email. Please sign up first.' });
  }

  // Compare password against stored hash
  const isMatch = await bcrypt.compare(password, user.password_hash);
  if (!isMatch) {
    console.log(`❌ Login failed – wrong password for: ${email}`);
    return res.status(401).json({ success: false, message: 'Invalid credentials. Please check your password. ❌' });
  }

  console.log(`✅ User logged in: ${email}`);

  return res.json({
    success: true,
    message: 'Login successful!',
    user: { id: user.id, email: user.email, name: user.name },
  });
});

// ── POST /api/forgot-password ─────────────────────────────────────────
app.post('/api/forgot-password', async (req, res) => {
  const { email } = req.body;

  if (!email) {
    return res.status(400).json({ success: false, message: 'Email is required.' });
  }

  // 1. Find user (generic response to avoid enumeration)
  const user = db.prepare('SELECT id FROM users WHERE email = ?').get(email.toLowerCase().trim());
  
  // Even if user doesn't exist, we send success to prevent email leakage
  const successResponse = { success: true, message: 'If an account exists with this email, a reset link has been sent.' };

  if (!user) {
    console.log(`[BeautifyAI] Forgot password request for non-existent email: ${email}`);
    return res.json(successResponse);
  }

  // 2. Generate 6-digit OTP
  const otp = Math.floor(100000 + Math.random() * 900000).toString();
  const hashedOtp = crypto.createHash('sha256').update(otp).digest('hex');
  const expiry = new Date(Date.now() + 15 * 60000).toISOString(); // 15 minutes

  // 3. Save to DB
  db.prepare('UPDATE users SET reset_token = ?, reset_token_expiry = ? WHERE id = ?')
    .run(hashedOtp, expiry, user.id);

  // 4. Send Email
  const mailOptions = {
    from: process.env.SMTP_FROM || '"BeautifyAI" <noreply@beautifyai.com>',
    to: email,
    subject: 'Your BeautifyAI Password Reset OTP',
    html: `
      <div style="font-family: sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #eee; border-radius: 10px;">
        <h2 style="color: #4f46e5;">BeautifyAI Password Reset</h2>
        <p>You requested a password reset for your account. Please use the 6-digit OTP below to securely reset your password. This OTP will expire in 15 minutes.</p>
        <div style="text-align: center; margin: 30px 0;">
          <span style="background-color: #f3f4f6; color: #111827; padding: 12px 24px; font-size: 24px; letter-spacing: 4px; border-radius: 8px; font-weight: bold; display: inline-block;">${otp}</span>
        </div>
        <hr style="border: 0; border-top: 1px solid #eee; margin: 20px 0;" />
        <p style="font-size: 12px; color: #999;">If you did not request this, you can safely ignore this email.</p>
      </div>
    `,
  };

  try {
    const info = await transporter.sendMail(mailOptions);
    console.log(`[BeautifyAI] Reset OTP sent to: ${email}`);
    if (!process.env.SMTP_USER) {
      console.log(`[BeautifyAI] 📨 View the test email here: ${nodemailer.getTestMessageUrl(info)}`);
    }
    return res.json({ success: true, message: 'If an account exists with this email, an OTP has been sent.', devOtp: otp });
  } catch (err) {
    console.error(`[BeautifyAI] ❌ Failed to send reset email to ${email}:`, err);
    // Log the OTP in dev mode for convenience
    console.log(`[BeautifyAI] Development OTP (Fallback): ${otp}`);
    
    return res.status(500).json({ 
      success: false, 
      message: 'Failed to send email. Check your SMTP configuration.',
      devOtp: otp 
    });
  }
});

// ── POST /api/verify-otp ───────────────────────────────────────
app.post('/api/verify-otp', async (req, res) => {
  const { email, otp } = req.body;

  if (!email || !otp || otp.length !== 6) {
    return res.status(400).json({ success: false, message: 'Invalid request.' });
  }

  const hashedOtp = crypto.createHash('sha256').update(otp).digest('hex');
  const user = db.prepare('SELECT id FROM users WHERE email = ? AND reset_token = ? AND reset_token_expiry > ?')
    .get(email.toLowerCase().trim(), hashedOtp, new Date().toISOString());

  if (!user) {
    return res.status(400).json({ success: false, message: 'Invalid or expired OTP.' });
  }

  return res.json({ success: true, message: 'OTP verified successfully.' });
});

// ── POST /api/reset-password ───────────────────────────────────
app.post('/api/reset-password', async (req, res) => {
  const { email, otp, password } = req.body;

  if (!password || password.length < 8) {
    return res.status(400).json({ success: false, message: 'Password must be at least 8 characters.' });
  }

  if (!email || !otp || otp.length !== 6) {
    return res.status(400).json({ success: false, message: 'Please enter a valid 6-digit OTP.' });
  }

  // 1. Hash the incoming OTP to match stored version
  const hashedOtp = crypto.createHash('sha256').update(otp).digest('hex');

  // 2. Find user with this OTP and ensure it's not expired
  const user = db.prepare('SELECT id FROM users WHERE email = ? AND reset_token = ? AND reset_token_expiry > ?')
    .get(email.toLowerCase().trim(), hashedOtp, new Date().toISOString());

  if (!user) {
    return res.status(400).json({ success: false, message: 'Invalid or expired OTP.' });
  }

  // 3. Update password and clear token
  const passwordHash = await bcrypt.hash(password, 12);
  db.prepare('UPDATE users SET password_hash = ?, reset_token = NULL, reset_token_expiry = NULL WHERE id = ?')
    .run(passwordHash, user.id);

  console.log(`[BeautifyAI] Password reset successful for user ID: ${user.id}`);
  
  return res.json({ success: true, message: 'Password has been reset successfully. You can now log in.' });
});

// ── Start Server ──────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`\n🚀 BeautifyAI Backend running at http://localhost:${PORT}`);
  console.log(`   Health check: http://localhost:${PORT}/api/health\n`);
});
