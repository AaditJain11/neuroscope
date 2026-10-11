/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        surface: {
          950: '#09090b',
          900: '#121215',
          850: '#18181c',
          800: '#27272a',
          700: '#3f3f46',
        },
        brand: {
          amber: '#f59e0b',
          accent: '#d97706',
          light: '#fef3c7',
        }
      }
    },
  },
  plugins: [],
}
