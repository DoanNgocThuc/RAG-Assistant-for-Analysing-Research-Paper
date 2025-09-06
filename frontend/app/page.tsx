// app/page.tsx
"use client";

import dynamic from "next/dynamic";
import React from "react";

const ResearchPaperChat = dynamic(() => import("@/components/ResearchPaperChat"), { ssr: false });

export default function Page() {
  return <ResearchPaperChat />;
}
