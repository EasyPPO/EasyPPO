# Iterate through all files matching 2-*.in / 2-*.out in the current directory
Get-ChildItem -Filter "2-*.in","2-*.out" | ForEach-Object {
    # Extract filename without extension
    $name = $_.BaseName
    $ext = $_.Extension

    # Use regex to match prefix like "2-01" or "2.11"
    if ($name -match '^2[-\.](\d+)$') {
        $num = [int]$matches[1]   # Extract number and convert to integer, e.g. 01 -> 1
        if ($ext -eq ".in") {
            $newName = "$num.in"
        } elseif ($ext -eq ".out") {
            $newName = "$num.ans"
        } else {
            return
        }
        Rename-Item $_.FullName -NewName $newName
    }
}
