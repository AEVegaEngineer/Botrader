import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { MantineProvider, ColorSchemeScript } from '@mantine/core';
import { theme } from '../lib/theme';
import { Navbar } from '../components/Navbar';
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Botrader",
  description: "Algorithmic Trading Bot Dashboard",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <ColorSchemeScript />
      </head>
      <body className={`${geistSans.variable} ${geistMono.variable}`}>
        <MantineProvider theme={theme} defaultColorScheme="dark">
          <Navbar />
          {children}
        </MantineProvider>
      </body>
    </html>
  );
}
