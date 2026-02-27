export const metadata = {
  title: 'Thus Spoke Robots Papers',
  description: 'Interactive paper comprehension quizzes',
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body style={{ margin: 0, background: '#0f172a', color: '#f8fafc' }}>
        {children}
      </body>
    </html>
  );
}
