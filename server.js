import express from "express";
import cors from "cors";

const app = express();

app.use(cors());
app.use(express.json());

app.get("/", (req, res) => {
  res.send("OK");
});

app.get("/api/health", (req, res) => {
  res.json({ ok: true });
});

app.post("/api/test", (req, res) => {
  res.json({ received: req.body });
});

const PORT = process.env.PORT || 3001;
app.listen(PORT, "0.0.0.0", () => {
  console.log("Server running on", PORT);
});
console.log("DASHSCOPE key exists:", !!process.env.DASHSCOPE_API_KEY);