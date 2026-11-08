for ($i = 6; $i -le 15; $i++) {
    New-Item -Path "$i.ans" -ItemType File -Force | Out-Null
}
