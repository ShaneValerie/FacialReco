<?php
header("Content-Type: application/json");
require "db.php";

$data = json_decode(file_get_contents("php://input"), true);

$stmt = $conn->prepare(
    "INSERT INTO attendance_log (employee_id, gate, direction, confidence, logged_at)
     VALUES (?, ?, ?, ?, NOW())"
);
$stmt->bind_param("issd", $data["employee_id"], $data["gate"], $data["direction"], $data["confidence"]);
$stmt->execute();

echo json_encode(["status" => "ok", "log_id" => $stmt->insert_id]);
?>
