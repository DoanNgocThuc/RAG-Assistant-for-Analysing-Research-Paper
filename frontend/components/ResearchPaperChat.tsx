// components/research/ResearchPaperChat.tsx
"use client";

import React from "react";
import { ChatProvider } from "./ChatProvider";
import { Header } from "./Header";
import { PDFCard } from "./PdfCard";
import { ChatCard } from "./ChatCard";
import { FormulaSheetAndDialog } from "./FormulaSheetAndDialog";

export const ResearchPaperChat: React.FC = () => {
  return (
    <ChatProvider>
      <div className="min-h-screen bg-background">
        <Header />
        <div className="container mx-auto px-3 py-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 max-h-[calc(100vh-6rem)]">
            <PDFCard />
            <ChatCard />
          </div>
        </div>
        <FormulaSheetAndDialog />
      </div>
    </ChatProvider>
  );
};

export default ResearchPaperChat;
