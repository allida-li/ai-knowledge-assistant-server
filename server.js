import express from "express";

const app = express();

app.get("/", (req, res) => {
  res.send("OK");
});

app.get("/api/health", (req, res) => {
  res.json({ ok: true });
});

const PORT = process.env.PORT || 3001;

app.listen(PORT, () => {
  console.log("Server running on", PORT);
});