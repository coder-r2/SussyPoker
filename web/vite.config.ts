import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// GitHub Pages serves the site from https://coder-r2.github.io/SussyPoker/
export default defineConfig({
  base: "/SussyPoker/",
  plugins: [react()],
});
