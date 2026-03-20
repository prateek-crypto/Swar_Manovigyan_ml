import type { EmotionQuadrant, EmotionInfo, Track } from "@/types";

export const EMOTION_MAP: Record<EmotionQuadrant, EmotionInfo> = {
  0: {
    label: "Sad / Melancholic",
    description:
      "Feeling sad, depressed, or melancholic. Gentle, soothing music can help.",
    color: "hsl(239, 84%, 67%)",
    icon: "CloudRain",
    gradient: "from-indigo-500/20 to-blue-500/20",
  },
  1: {
    label: "Calm / Peaceful",
    description:
      "Feeling calm, peaceful, or relaxed. Perfect for meditation or quiet moments.",
    color: "hsl(142, 71%, 45%)",
    icon: "Leaf",
    gradient: "from-green-500/20 to-emerald-500/20",
  },
  2: {
    label: "Angry / Stressed",
    description:
      "Feeling angry, stressed, or anxious. Energetic music can channel these emotions.",
    color: "hsl(0, 84%, 60%)",
    icon: "Flame",
    gradient: "from-red-500/20 to-orange-500/20",
  },
  3: {
    label: "Happy / Excited",
    description:
      "Feeling happy, excited, or energetic. Great for dancing and celebrating!",
    color: "hsl(38, 92%, 50%)",
    icon: "Sun",
    gradient: "from-amber-500/20 to-yellow-500/20",
  },
};

export const FALLBACK_RECOMMENDATIONS: Record<EmotionQuadrant, string[]> = {
  0: [
    "Sad ballads",
    "Slow acoustic songs",
    "Melancholic classical music",
    "Indie folk",
    "Blues",
    "Ambient music",
    "Piano solos",
  ],
  1: [
    "Meditation music",
    "Nature sounds",
    "Soft jazz",
    "Ambient",
    "Classical music",
    "Folk music",
    "Acoustic guitar",
  ],
  2: [
    "Heavy metal",
    "Punk rock",
    "Hard rock",
    "Rap",
    "Electronic",
    "Industrial",
    "Alternative rock",
  ],
  3: [
    "Pop music",
    "Dance music",
    "Upbeat rock",
    "Funk",
    "Disco",
    "Reggae",
    "Ska",
  ],
};

export const FALLBACK_TRACKS: Record<EmotionQuadrant, Track[]> = {
  0: [
    { id: "sad1", title: "Hurt", artist: "Johnny Cash", duration: "3:38" },
    { id: "sad2", title: "Mad World", artist: "Gary Jules", duration: "3:09" },
    {
      id: "sad3",
      title: "The Sound of Silence",
      artist: "Simon & Garfunkel",
      duration: "3:05",
    },
  ],
  1: [
    {
      id: "calm1",
      title: "Weightless",
      artist: "Marconi Union",
      duration: "8:59",
    },
    {
      id: "calm2",
      title: "Clair de Lune",
      artist: "Claude Debussy",
      duration: "4:56",
    },
    {
      id: "calm3",
      title: "Teardrop",
      artist: "Massive Attack",
      duration: "5:30",
    },
  ],
  2: [
    {
      id: "angry1",
      title: "Break Stuff",
      artist: "Limp Bizkit",
      duration: "2:46",
    },
    {
      id: "angry2",
      title: "Killing in the Name",
      artist: "Rage Against the Machine",
      duration: "5:13",
    },
    {
      id: "angry3",
      title: "Bodies",
      artist: "Drowning Pool",
      duration: "3:21",
    },
  ],
  3: [
    {
      id: "happy1",
      title: "Happy",
      artist: "Pharrell Williams",
      duration: "3:53",
    },
    {
      id: "happy2",
      title: "Don't Stop Me Now",
      artist: "Queen",
      duration: "3:29",
    },
    {
      id: "happy3",
      title: "Good Vibrations",
      artist: "The Beach Boys",
      duration: "3:37",
    },
  ],
};

export const AUDIO_FEATURE_RANGES = {
  acousticness: { min: 0, max: 1, step: 0.01, default: 0.5 },
  danceability: { min: 0, max: 1, step: 0.01, default: 0.5 },
  energy: { min: 0, max: 1, step: 0.01, default: 0.5 },
  instrumentalness: { min: 0, max: 1, step: 0.01, default: 0.0 },
  liveness: { min: 0, max: 1, step: 0.01, default: 0.1 },
  loudness: { min: -60, max: 0, step: 1, default: -20 },
  speechiness: { min: 0, max: 1, step: 0.01, default: 0.1 },
  tempo: { min: 50, max: 200, step: 1, default: 120 },
  valence: { min: 0, max: 1, step: 0.01, default: 0.5 },
} as const;

export const FEATURE_LABELS: Record<string, string> = {
  acousticness: "Acousticness",
  danceability: "Danceability",
  energy: "Energy",
  instrumentalness: "Instrumentalness",
  liveness: "Liveness",
  loudness: "Loudness (dB)",
  speechiness: "Speechiness",
  tempo: "Tempo (BPM)",
  valence: "Valence",
};
