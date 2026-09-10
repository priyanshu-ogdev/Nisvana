import js from "@eslint/js";

export default [
    js.configs.recommended,
    {
        files: ["src/**/*.js", "src/**/*.jsx"],
        languageOptions: {
            ecmaVersion: "latest",
            sourceType: "module",
            parserOptions: {
                ecmaFeatures: { jsx: true }
            },
            globals: {
                window: "readonly",
                document: "readonly",
                requestAnimationFrame: "readonly",
                cancelAnimationFrame: "readonly",
                performance: "readonly",
                console: "readonly",
                setTimeout: "readonly",
                clearTimeout: "readonly",
                Math: "readonly",
                Float32Array: "readonly",
                setInterval: "readonly",
                clearInterval: "readonly",
                WebSocket: "readonly"
            }
        },
        rules: {
            "no-unused-vars": "off"
        }
    }
];
