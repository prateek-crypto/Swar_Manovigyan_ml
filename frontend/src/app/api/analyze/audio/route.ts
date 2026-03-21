import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();
    const backendFormData = new FormData();

    const file = formData.get("file") ?? formData.get("audio");
    if (!(file instanceof Blob)) {
      return NextResponse.json(
        { error: "Audio file is required" },
        { status: 400 }
      );
    }

    const fileName = file instanceof File ? file.name : "upload.wav";
    backendFormData.append("file", file, fileName);

    const backendRes = await fetch(`${BACKEND_URL}/analyze/upload`, {
      method: "POST",
      body: backendFormData,
    });

    if (!backendRes.ok) {
      return NextResponse.json(
        { error: "Backend audio analysis failed" },
        { status: backendRes.status }
      );
    }

    const data = await backendRes.json();
    return NextResponse.json(data);
  } catch {
    return NextResponse.json(
      { error: "Failed to connect to backend" },
      { status: 502 }
    );
  }
}
