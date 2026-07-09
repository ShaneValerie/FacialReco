<?php
$conn = new mysqli("localhost", "root", "", "tenant_mgmt");
if ($conn->connect_error) {
    http_response_code(500);
    die(json_encode(["error" => "DB connection failed"]));
}
?>


