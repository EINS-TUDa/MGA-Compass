import sharp from 'sharp'
import { mkdir } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import path from 'node:path'

const here = path.dirname(fileURLToPath(import.meta.url))
const outDir = path.join(here, '..', 'public')

const compassPaths = `
  <g transform="rotate(-15, 12, 12)">
    <path d="M6 6.75L20 12L6 17.25Z" fill="#a8e6cf"/>
    <path d="M6 5.5V18.5" stroke="#000000" stroke-width="0.2" stroke-linecap="round"/>
    <path d="M4 6L20 12L4 18" stroke="#000000" stroke-width="0.2" stroke-linecap="round" stroke-linejoin="round"/>
    <path
      d="M12,2A10,10 0 0,0 2,12A10,10 0 0,0 12,22A10,10 0 0,0 22,12A10,10 0 0,0 12,2M12,4A8,8 0 0,1 20,12A8,8 0 0,1 12,20A8,8 0 0,1 4,12A8,8 0 0,1 12,4M17,7L13.5,13.5L7,17L10.5,10.5L17,7M12,11A1,1 0 0,0 11,12A1,1 0 0,0 12,13A1,1 0 0,0 13,12A1,1 0 0,0 12,11Z"
      fill="#485fc7"
      transform="translate(10, 12) scale(0.32) translate(-12, -12)"
    />
  </g>
`

const standardSvg = `<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
  <rect width="24" height="24" fill="#ffffff"/>
  ${compassPaths}
</svg>`

// Maskable icons need the artwork inside the center ~80% safe zone, so pad the canvas.
const maskableSvg = `<svg viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
  <rect width="32" height="32" fill="#ffffff"/>
  <g transform="translate(4, 4)">${compassPaths}</g>
</svg>`

await mkdir(outDir, { recursive: true })

const targets = [
  { svg: standardSvg, size: 192, file: 'pwa-192x192.png' },
  { svg: standardSvg, size: 512, file: 'pwa-512x512.png' },
  { svg: standardSvg, size: 180, file: 'apple-touch-icon.png' },
  { svg: maskableSvg, size: 512, file: 'maskable-icon-512x512.png' },
]

for (const { svg, size, file } of targets) {
  await sharp(Buffer.from(svg), { density: 384 })
    .resize(size, size)
    .png()
    .toFile(path.join(outDir, file))
  console.log('wrote', file)
}
