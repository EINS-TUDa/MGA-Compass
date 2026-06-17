// Round to 1 decimal, dropping a trailing ".0" (used for percentage display).
export const formatPercent = (value) => {
  const rounded = Math.round(value * 10) / 10
  return Number.isInteger(rounded) ? String(rounded) : rounded.toFixed(1)
}

// Format a number using k/M/B suffixes once its magnitude crosses 1000.
// `referenceAbs` lets callers pick the divisor based on a different value
// (e.g. the max of a range) than the one being formatted.
export const formatCompact = (value, referenceAbs = Math.abs(value)) => {
  let divisor = 1
  let suffix = ''
  if (referenceAbs >= 1_000_000_000) { divisor = 1_000_000_000; suffix = 'B' }
  else if (referenceAbs >= 1_000_000) { divisor = 1_000_000; suffix = 'M' }
  else if (referenceAbs >= 1_000) { divisor = 1_000; suffix = 'k' }

  return (value / divisor).toFixed(2).replace(/\.?0+$/, '') + suffix
}
