module.exports = {
    content: [
        "./index.html",
        "./src/**/*.{ts,tsx}"
    ],
    theme: {
        extend: {
            colors: {
                graphite: {
                    950: "#080c10",
                    900: "#0c141a",
                    800: "#121d25",
                    700: "#18252f",
                    600: "#223241",
                    500: "#2f4254"
                },
                accent: {
                    primary: "#2fb8ff",
                    secondary: "#5fd3ff",
                    success: "#3bd989",
                    warning: "#ffc857",
                    error: "#ff5c6f"
                },
                warning: "#ffc857",
                success: "#3bd989",
                error: "#ff5c6f",
                haze: "#93a5b5",
                bone: "#e8f0f6"
            },
            fontFamily: {
                display: ["\"Sora\"", "sans-serif"],
                mono: ["\"JetBrains Mono\"", "monospace"]
            },
            boxShadow: {
                card: "0 24px 40px rgba(6, 12, 18, 0.55)",
                accent: "0 0 0 1px rgba(47, 184, 255, 0.35), 0 0 18px rgba(47, 184, 255, 0.25)"
            },
            letterSpacing: {
                brutal: "0.35em"
            }
        },
    },
    plugins: [],
};
