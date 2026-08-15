on titleMatches(theTitle, theList)
  repeat with frag in theList
    if theTitle contains frag then return true
  end repeat
  return false
end titleMatches

set allowTitles to {"wants to access", "would like to access", "wants to control", "wants to use", "would like to receive keystrokes", "access data from other apps"}
set denyTitles to {"sudo", "ssh", "Keychain", "keychain", "wallet", "Wallet", "Touch ID", "Password"}
-- Cancel-list: admin-elevation prompts that should be auto-DISMISSED (NEVER granted).
-- These are security-boundary requests. We cancel them so they stop blocking the user
-- while leaving the calling process to handle the rejection. Includes the python3.11
-- "would like to administer your computer" SecurityAgent prompt observed 2026-05-20.
set cancelTitles to {"would like to administer your computer", "wants to administer your computer", "administer your computer"}
set results to ""
-- SecurityAgent hosts the admin auth dialog; add it alongside the existing TCC apps.
set tccApps to {"UserNotificationCenter", "usernoted", "NotificationCenter", "NotificationCenterUI", "SystemUIServer", "coreservicesd", "tccd", "SecurityAgent", "CoreServicesUIAgent"}

tell application "System Events"
  repeat with appName in tccApps
    try
      if exists application process appName then
        tell application process appName
          repeat with w in windows
            try
              set winTitle to (title of w as string)
            on error
              set winTitle to ""
            end try
            if winTitle is not "" then
              if my titleMatches(winTitle, cancelTitles) then
                try
                  click button "Cancel" of w
                  set results to results & "cancelled_admin|" & winTitle & "|" & appName & linefeed
                on error
                  try
                    -- fallback: try "Don't Allow" or press Escape key
                    click button "Don't Allow" of w
                    set results to results & "cancelled_admin|" & winTitle & "|" & appName & linefeed
                  on error
                    set results to results & "stuck_admin|" & winTitle & "|" & appName & linefeed
                  end try
                end try
              else if my titleMatches(winTitle, denyTitles) then
                set results to results & "skipped_deny|" & winTitle & "|" & appName & linefeed
              else if my titleMatches(winTitle, allowTitles) then
                try
                  click button "Allow" of w
                  set results to results & "clicked|" & winTitle & "|" & appName & linefeed
                on error
                  set results to results & "stuck|" & winTitle & "|" & appName & linefeed
                end try
              end if
            end if
          end repeat
        end tell
      end if
    on error errMsg
      -- skip
    end try
  end repeat
end tell

return results
